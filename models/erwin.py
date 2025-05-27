from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from einops import rearrange, reduce

from typing import Literal, List
from dataclasses import dataclass

from balltree import build_balltree

from gatr.layers import (
    EquiLayerNorm,
    EquiLinear,
    GeoMLP,
    SelfAttention,
    SelfAttentionConfig,
    ScalarGatedNonlinearity,
)
from gatr.layers.mlp import MLPConfig, ScalarGatedNonlinearity
from gatr.interface import embed_point, embed_translation
from gatr.utils.tensors import construct_reference_multivector

from .mpnn_variants import MPNN, DistanceBasedScalarOnlyMPNN
from .ball_pooling_unpooling_variants import (
    BallPoolingRelDist,
    BallUnpoolingRelDist,
    BallPoolingRelDistRelPosMv,
    BallUnpoolingRelDistRelPosMv,
    Node,
)


class ErwinEmbedding(nn.Module):
    """Linear projection -> MPNN."""

    def __init__(
        self,
        in_dim: int,
        dim: int,
        mp_steps: int,
        dimensionality: int = 3,
        mpnn_type: str = "scalar_only"  # New parameter to specify MPNN type
    ):
        super().__init__()
        self.mp_steps = mp_steps
        # in_mv is 1 channel (from embed_point). in_s is in_dim (e.g., 16) channels.
        self.embed_fn = EquiLinear(
            in_mv_channels=in_dim,
            out_mv_channels=dim,
            in_s_channels=in_dim,
            out_s_channels=dim,
        )

        # Select MPNN type based on parameter
        if mpnn_type == "scalar_only":
            self.mpnn = DistanceBasedScalarOnlyMPNN(dim, mp_steps, mlp_ratio=2)
        else: # mpnn_type == "original" which is using both scalar and multivector MPNN, 3 times slower both more accurate
            self.mpnn = MPNN(dim, mp_steps, 16)  # Original MPNN as fallback

    def forward(
        self,
        mv: torch.Tensor,
        sc: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        mv, sc = self.embed_fn(mv, sc)
        if isinstance(self.mpnn, DistanceBasedScalarOnlyMPNN):
            # For scalar-only MPNN, we only pass and return scalar features
            sc = self.mpnn(sc, pos, edge_index) if self.mp_steps > 0 else sc
            return mv, sc  # mv passes through unchanged
        else:
            # Original MPNN behavior
            return self.mpnn(mv, sc, pos, edge_index) if self.mp_steps > 0 else (mv, sc)


class BallMSA(nn.Module):
    """
    Ball Multi-Head Self-Attention (BMSA) module (eq. 8).
    """

    def __init__(
        self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 16, use_distance_bias: bool = False
    ): # dimensionality here refers to the GA's dimension, typically 16 for G(3,0,1)
        super().__init__()
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.feature_dim = dim
        self.use_distance_bias = use_distance_bias

        # Add the sigma parameter for distance-based attention bias only if needed
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1))) if use_distance_bias else None

        # GATr's SelfAttention with distance-based attention bias
        attention_config = SelfAttentionConfig(
            num_heads=self.num_heads,
            multi_query=False, # As per original snippet
            in_mv_channels=self.feature_dim,
            out_mv_channels=self.feature_dim,
            in_s_channels=self.feature_dim,
            out_s_channels=self.feature_dim,
        )
        self.attention = SelfAttention(attention_config)

        # Final output projection layer
        self.projection = EquiLinear(
            in_mv_channels=self.feature_dim,
            out_mv_channels=self.feature_dim,
            in_s_channels=self.feature_dim,
            out_s_channels=self.feature_dim,
        )

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """ Distance-based attention bias (eq. 10). """
        if not self.use_distance_bias:
            return None

        pos = rearrange(pos, '(n m) d -> n m d', m=self.ball_size)
        # Create attention mask based on pairwise distances
        attention_bias = self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)
        # Convert to attention mask format expected by GATr
        return attention_bias

    def forward(self, mv: torch.Tensor, sc: torch.Tensor, pos: torch.Tensor):
        # mv shape: (N_total, feature_dim, algebra_dim), e.g., (B*S, C, 16)
        # sc shape: (N_total, feature_dim), e.g., (B*S, C)
        # pos shape: (N_total, 3) - currently unused in this simplified version
        # N_total = num_balls * ball_size

        N_total = mv.shape[0]
        num_balls = N_total // self.ball_size

        # Reshape for per-ball attention:
        # (num_balls, ball_size, feature_dim, algebra_dim)
        mv_reshaped = rearrange(mv, '(n m) c d -> n m c d', n=num_balls, m=self.ball_size)
        # (num_balls, ball_size, feature_dim)
        sc_reshaped = rearrange(sc, '(n m) c -> n m c', n=num_balls, m=self.ball_size)

        # Create attention mask based on distances only if use_distance_bias is True
        attention_mask = self.create_attention_mask(pos) if self.use_distance_bias else None

        # Apply GATr's SelfAttention per ball with the distance-based attention mask
        mv_attended, sc_attended = self.attention(
            multivectors=mv_reshaped,
            scalars=sc_reshaped,
            attention_mask=attention_mask
        )
        # mv_attended shape: (num_balls, ball_size, feature_dim, algebra_dim)
        # sc_attended shape: (num_balls, ball_size, feature_dim)

        # Reshape back to original flat structure
        mv_out = rearrange(mv_attended, 'n m c d -> (n m) c d')
        sc_out = rearrange(sc_attended, 'n m c -> (n m) c')

        # Apply the final output projection
        return self.projection(mv_out, sc_out)


class UpdateModule(nn.Module):
    def __init__(self, dim_channel: int):
        super().__init__()
        self.equi_linear = EquiLinear(
            in_mv_channels=2 * dim_channel,
            out_mv_channels=dim_channel,
            in_s_channels=2 * dim_channel,
            out_s_channels=dim_channel
        )
        self.norm = EquiLayerNorm()

    def forward(self, mv_in: torch.Tensor, sc_in: torch.Tensor):
        mv_processed, sc_processed = self.equi_linear(mv_in, sc_in)
        return self.norm(mv_processed, sc_processed)


class ErwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        dimensionality: int = 16,
        use_distance_bias: bool = False,
    ):
        super().__init__()
        self.ball_size = ball_size

        # EquiLayerNorm will handle both mv and sc # TODO: check the mv_channel_dim=-2 thing
        # -2 is default, so shouldn't need specification or??
        self.norm1 = EquiLayerNorm(
            mv_channel_dim=-2
        )  # mv shape (..., channels, 16) and sc shape (..., channels)
        self.norm2 = EquiLayerNorm(mv_channel_dim=-2)

        self.BMSA = BallMSA(dim, num_heads, ball_size, dimensionality, use_distance_bias=use_distance_bias)

        self.geo_mlp = GeoMLP(
            MLPConfig(
                mv_channels=[dim, dim * mlp_ratio, dim],
                s_channels=[dim, dim * mlp_ratio, dim],
                activation="gelu",
            )
        )

    def forward(self, mv: torch.Tensor, sc: torch.Tensor, pos: torch.Tensor, reference_mv: torch.Tensor):

        # Store original for residual connection
        mv_residual, sc_residual = mv, sc

        # BallMSA.forward is (mv, sc, pos)
        mv, sc = self.BMSA(*self.norm1(mv, sc), pos)

        # First residual connection
        mv, sc = mv + mv_residual, sc + sc_residual

        # Store for second residual connection
        mv_residual, sc_residual = mv, sc

        # GeoMPL.forward is (mv, sc)
        mv, sc = self.geo_mlp(*self.norm2(mv, sc), reference_mv=reference_mv)

        # Second residual connection
        return mv + mv_residual, sc + sc_residual


class BasicLayer(nn.Module):
    def __init__(
        self,
        direction: Literal[
            "down", "up", None
        ],  # down: encoder, up: decoder, None: bottleneck
        depth: int,
        stride: int | None,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        rotate: bool,
        dimensionality: int = 3, # Spatial dimensionality for embed_translation if used
        algebra_dimensionality: int = 16, # GA dimensionality for blocks
        use_distance_bias: bool = False,
        pooling_type: str = "RelDistRelPosMv", # New parameter
        unpooling_type: str = "RelDistRelPosMv", # New parameter
    ):
        super().__init__()
        hidden_dim = in_dim if direction == "down" else out_dim

        # No. of TransformerBlocks = depth
        self.blocks = nn.ModuleList(
            [
                ErwinTransformerBlock(
                    hidden_dim, num_heads, ball_size, mlp_ratio, algebra_dimensionality, use_distance_bias=use_distance_bias
                )
                for _ in range(depth)
            ]
        )
        self.rotate = [i % 2 for i in range(depth)] if rotate else [False] * depth

        self.pool = lambda node: node
        self.unpool = lambda node: node

        # Select pooling strategy
        if direction == "down" and stride is not None:
            if pooling_type == "RelDist":
                self.pool = BallPoolingRelDist(hidden_dim, out_dim, stride, dimensionality)
            elif pooling_type == "RelDistRelPosMv":
                self.pool = BallPoolingRelDistRelPosMv(hidden_dim, out_dim, stride, dimensionality)
            else:
                raise ValueError(f"Unknown pooling_type: {pooling_type}")

        # Select unpooling strategy
        elif direction == "up" and stride is not None:
            if unpooling_type == "RelDist":
                self.unpool = BallUnpoolingRelDist(in_dim, hidden_dim, stride, dimensionality)
            elif unpooling_type == "RelDistRelPosMv":
                self.unpool = BallUnpoolingRelDistRelPosMv(in_dim, hidden_dim, stride, dimensionality)
            else:
                raise ValueError(f"Unknown unpooling_type: {unpooling_type}")

    def forward(self, node: Node, reference_mv: torch.Tensor) -> Node:
        node = self.unpool(node)
        for blk in self.blocks:
            # Process the node through the block
            mv, sc = blk(node.mv, node.sc, node.pos, reference_mv=reference_mv)
            node.mv = mv
            node.sc = sc

        return self.pool(node)


class ErwinTransformer(nn.Module):
    """
    Erwin Transformer.

    Args:
        c_in (int): number of input channels.
        c_hidden (List): number of hidden channels for each encoder + bottleneck layer (reverse for decoder).
        ball_size (List): list of ball sizes for each encoder layer (reverse for decoder).
        enc_num_heads (List): list of number of heads for each encoder layer.
        enc_depths (List): list of number of ErwinTransformerBlock layers for each encoder layer.
        dec_num_heads (List): list of number of heads for each decoder layer.
        dec_depths (List): list of number of ErwinTransformerBlock layers for each decoder layer.
        strides (List): list of strides for each encoder layer (reverse for decoder).
        rotate (int): angle of rotation for cross-ball interactions; if 0, no rotation.
        decode (bool): whether to decode or not. If not, returns latent representation at the coarsest level.
        mlp_ratio (int): ratio of GeoMLP's hidden dim to a layer's hidden dim.
        dimensionality (int): spatial dimensionality of the input data (e.g., 3 for 3D points).
        algebra_dimensionality (int): dimensionality of the geometric algebra (e.g., 16 for G(3,0,1)).
        mp_steps (int): number of message passing steps in the MPNN Embedding.
        use_distance_bias (bool): whether to use distance-based attention bias.
        mpnn_type (str): type of MPNN to use.
        pooling_type (str): type of pooling to use ("RelDist" or "RelDistRelPosMv").
        unpooling_type (str): type of unpooling to use ("RelDist" or "RelDistRelPosMv").

    Notes:
        - lengths of ball_size, enc_num_heads, enc_depths must be the same N (as it includes encoder and bottleneck).
        - lengths of strides, dec_num_heads, dec_depths must be N - 1.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: list[int],
        ball_sizes: List,
        enc_num_heads: List,
        enc_depths: List,
        dec_num_heads: List,
        dec_depths: List,
        strides: List,
        rotate: int,
        decode: bool = True,
        mlp_ratio: int = 4,
        dimensionality: int = 3, # Spatial dimensionality
        algebra_dimensionality: int = 16, # GA dimensionality
        mp_steps: int = 3,
        use_distance_bias: bool = False,
        mpnn_type: str = "scalar_only",
        pooling_type: str = "RelDistRelPosMv", # New parameter
        unpooling_type: str = "RelDistRelPosMv", # New parameter
    ):
        super().__init__()
        assert len(enc_num_heads) == len(enc_depths) == len(ball_sizes)
        assert len(dec_num_heads) == len(dec_depths) == len(strides)
        assert len(strides) == len(ball_sizes) - 1

        self.rotate = rotate
        self.decode = decode
        self.ball_sizes = ball_sizes
        self.strides = strides

        self.embed = ErwinEmbedding(
            in_dim=c_in,
            dim=c_hidden[0],
            mp_steps=mp_steps,
            dimensionality=algebra_dimensionality, # ErwinEmbedding expects GA dimensionality
            mpnn_type=mpnn_type
        )

        num_layers = len(enc_depths) - 1  # last one is a bottleneck

        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(
                BasicLayer(
                    direction="down",
                    depth=enc_depths[i],
                    stride=strides[i],
                    in_dim=c_hidden[i],
                    out_dim=c_hidden[i + 1],
                    num_heads=enc_num_heads[i],
                    ball_size=ball_sizes[i],
                    rotate=rotate > 0,
                    mlp_ratio=mlp_ratio,
                    dimensionality=dimensionality, # Pass spatial dimensionality
                    algebra_dimensionality=algebra_dimensionality, # Pass GA dimensionality
                    use_distance_bias=use_distance_bias,
                    pooling_type=pooling_type, # Pass pooling_type
                    unpooling_type=unpooling_type, # Pass unpooling_type
                )
            )

        self.bottleneck = BasicLayer(
            direction=None,
            depth=enc_depths[-1],
            stride=None,
            in_dim=c_hidden[-1],
            out_dim=c_hidden[-1],
            num_heads=enc_num_heads[-1],
            ball_size=ball_sizes[-1],
            rotate=rotate > 0,
            mlp_ratio=mlp_ratio,
            dimensionality=dimensionality,
            algebra_dimensionality=algebra_dimensionality,
            use_distance_bias=use_distance_bias,
            # Bottleneck doesn't pool/unpool, so types are not strictly needed but pass for consistency
            pooling_type=pooling_type,
            unpooling_type=unpooling_type,
        )

        if decode:
            self.decoder = nn.ModuleList()
            for i in range(num_layers - 1, -1, -1):
                self.decoder.append(
                    BasicLayer(
                        direction="up",
                        depth=dec_depths[i],
                        stride=strides[i],
                        in_dim=c_hidden[i + 1],
                        out_dim=c_hidden[i],
                        num_heads=dec_num_heads[i],
                        ball_size=ball_sizes[i],
                        rotate=rotate > 0,
                        mlp_ratio=mlp_ratio,
                        dimensionality=dimensionality,
                        algebra_dimensionality=algebra_dimensionality,
                        use_distance_bias=use_distance_bias,
                        pooling_type=pooling_type, # Pass pooling_type
                        unpooling_type=unpooling_type, # Pass unpooling_type
                    )
                )

        self.in_dim = c_in
        self.out_dim = c_hidden[0]
        # Pass spatial dimensionality, used by pooling/unpooling if they embed positions
        self.dimensionality = dimensionality
        self.apply(self._init_weights)

    # No need to initialize weights of nn.Linaer, nn.LayerNorm
    # Initialization of EquiLinaer is handled by GATr
    # Unlike nn.LayerNorm, no parameters to initialize in EquiLayerNorm
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        node_features_mv: torch.Tensor,
        node_features_sc: torch.Tensor,
        node_positions: torch.Tensor,
        batch_idx: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        tree_idx: torch.Tensor | None = None,
        tree_mask: torch.Tensor | None = None,
        radius: float | None = None,
        **kwargs,
    ):
        with torch.no_grad():
            # TODO: check if the edge_index calculation is optimal, do we redo it every time?
            # if not given, build the ball tree and radius graph
            if tree_idx is None and tree_mask is None:
                tree_idx, tree_mask = build_balltree(
                    node_positions,
                    batch_idx,
                )
            if edge_index is None and self.embed.mp_steps:
                assert (
                    radius is not None
                ), "radius (float) must be provided if edge_index is not given to build radius graph"
                edge_index = torch_cluster.radius_graph(
                    node_positions, radius, batch=batch_idx, loop=True
                )

        self.reference_mv = construct_reference_multivector('data', node_features_mv)
        mv, sc = self.embed(
            node_features_mv, node_features_sc, node_positions, edge_index
        )

        node = Node(
            mv=mv[tree_idx],
            sc=sc[tree_idx],
            pos=node_positions[tree_idx],
            batch_idx=batch_idx[tree_idx],
            tree_idx_rot=None,  # will be populated in the encoder
        )

        for layer in self.encoder:
            node = layer(node, self.reference_mv)

        node = self.bottleneck(node, self.reference_mv)

        if self.decode:
            for layer in self.decoder:
                node = layer(node, self.reference_mv)
            return (
                node.mv[tree_mask][torch.argsort(tree_idx[tree_mask])],
                node.sc[tree_mask][torch.argsort(tree_idx[tree_mask])],
            )

        return node.mv, node.sc, node.batch_idx
