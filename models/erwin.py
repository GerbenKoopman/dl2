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
from gatr.interface import embed_point
from gatr.utils.tensors import construct_reference_multivector


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, num_receivers: int):
    """
    Averages all values from src into the receivers at the indices specified by idx.
    Handles both 2D (scalar) and 3D (multivector) src tensors.

    Args:
        src (torch.Tensor): Source tensor.
                            For scalars: (N_messages, D_features).
                            For multivectors: (N_messages, D_channels, D_algebra).
        idx (torch.Tensor): Indices tensor of shape (N_messages,).
        num_receivers (int): Number of receivers.

    Returns:
        torch.Tensor: Result tensor.
                      For scalars: (num_receivers, D_features).
                      For multivectors: (num_receivers, D_channels, D_algebra).
    """
    if src.ndim == 2:  # Scalar case
        # src shape: (N_messages, D_features)
        # result shape: (num_receivers, D_features)
        result = torch.zeros(num_receivers, src.size(1), dtype=src.dtype, device=src.device)
        count_unsqueeze_dims = 1
    elif src.ndim == 3:  # Multivector case
        # src shape: (N_messages, D_channels, D_algebra)
        # result shape: (num_receivers, D_channels, D_algebra)
        result = torch.zeros(num_receivers, src.size(1), src.size(2), dtype=src.dtype, device=src.device)
        count_unsqueeze_dims = 2
    else:
        raise ValueError(f"Unsupported src ndim: {src.ndim}. Shape was {src.shape}")

    count = torch.zeros(num_receivers, dtype=torch.long, device=src.device)
    
    result.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))

    clamped_count = count.clamp(min=1)
    if count_unsqueeze_dims == 1:
        clamped_count = clamped_count.unsqueeze(1)
    elif count_unsqueeze_dims == 2:
        clamped_count = clamped_count.unsqueeze(1).unsqueeze(2)
    
    return result / clamped_count


class MPNN(nn.Module):
    """
    Message Passing Neural Network (see Gilmer et al., 2017).
        m_ij = MLP([h_i, h_j, pos_i - pos_j])       message
        m_i = mean(m_ij)                            aggregate
        h_i' = MLP([h_i, m_i])                      update

    """

    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3, mlp_ratio: int = 2):
        super().__init__()


        self.reference_mv = construct_reference_multivector(
            "canonical", torch.ones(dimensionality)
        )

        self.message_fns = nn.ModuleList(
            [
                GeoMLP(
                    MLPConfig(
                        mv_channels=[2*dim, 2 * mlp_ratio * dim, dim],
                        s_channels=[2*dim + 1, 2 * mlp_ratio * dim , dim], # +1 for the relative distance
                        activation="gelu",

                    )
        )
                for _ in range(mp_steps)
            ]
        )

        self.update_fns = nn.ModuleList(
            [
                UpdateModule(dim)
                for _ in range(mp_steps)
            ]
        )

    def layer(
        self, 
        message_fn: nn.Module,
        update_fn: nn.Module,
        mv: torch.Tensor,
        sc: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        row, col = edge_index
        
        # Construct message inputs by concatenating source and target features
        # For multivectors: [mv_i, mv_j]
        # For scalars: [sc_i, sc_j, edge_attr]
        
        # mv shape: (N_nodes, N_channels, D_algebra) -> mv[row]: (N_edges, N_channels, D_algebra)
        # Concatenate along N_channels dim (dim=1) - NOTE: dim=-2 would work too
        mv_msg_input = torch.cat([mv[row], mv[col]], dim=1)
        
        # sc shape: (N_nodes, N_channels_scalar) -> sc[row]: (N_edges, N_channels_scalar)
        # edge_attr shape: (N_edges, 1)
        # Concatenate along N_channels_scalar dim (dim=-1)
        sc_msg_input = torch.cat([sc[row], sc[col], edge_attr], dim=-1)
        
        # Compute messages using GeoMLP
        mv_messages, sc_messages = message_fn(mv_msg_input, sc_msg_input, self.reference_mv.to(mv.device)) # TODO: reference_mv?
        
        # Aggregate messages per receiver node (col)
        mv_agg = scatter_mean(mv_messages, col, mv.size(0))
        sc_agg = scatter_mean(sc_messages, col, sc.size(0))
        
        # Update node features
        # mv_input_for_update: (N_nodes, 2 * N_channels, D_algebra)
        # sc_input_for_update: (N_nodes, 2 * N_channels_scalar)
        mv_update, sc_update = update_fn(
            torch.cat([mv, mv_agg], dim=1), # NOTE: dim=-2 would work too
            torch.cat([sc, sc_agg], dim=-1)
        )
        
        # Residual connection
        return mv + mv_update, sc + sc_update

    @torch.no_grad()
    def compute_edge_attr(self, pos, edge_index):
        # Ensure edge_attr is (num_edges, 1) for concatenation
        return torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1, keepdim=True)

    def forward(
        self,
        mv: torch.Tensor,
        sc: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        edge_attr = self.compute_edge_attr(pos, edge_index)
        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            mv, sc = self.layer(message_fn, update_fn, mv, sc, edge_attr, edge_index)
            
        return mv, sc


class ErwinEmbedding(nn.Module):
    """Linear projection -> MPNN."""

    def __init__(self, in_dim: int, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.mp_steps = mp_steps
        # in_mv is 1 channel (from embed_point). in_s is in_dim (e.g., 16) channels.
        self.embed_fn = EquiLinear(
            in_mv_channels=1,
            out_mv_channels=dim,
            in_s_channels=in_dim,
            out_s_channels=dim,
        )
        self.mpnn = MPNN(dim, mp_steps, 16)

    def forward(
        self,
        mv: torch.Tensor,
        sc: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        mv, sc = self.embed_fn(mv, sc)
        return self.mpnn(mv, sc, pos, edge_index) if self.mp_steps > 0 else (mv, sc)


@dataclass
class Node:
    """Dataclass to store the hierarchical node information."""

    mv: torch.Tensor
    sc: torch.Tensor
    pos: torch.Tensor
    batch_idx: torch.Tensor
    tree_idx_rot: torch.Tensor | None = None
    children: Node | None = None


class BallPooling(nn.Module):
    """
    Pooling of leaf nodes in a ball:
        1. select balls of size 'stride'.
        2. concatenate leaf nodes inside each ball along with their relative positions to the ball center.
        3. apply equilinear projection and normalization.
        4. the output is the center of each ball endowed with the pooled features.
    """

    def __init__(
        self, in_dim: int, out_dim: int, stride: int = 2, dimensionality: int = 3
    ):
        super().__init__()
        self.stride = stride

        # Single EquiLinear for both multivectors and scalars
        self.projection = EquiLinear(
            in_mv_channels=in_dim * stride,
            out_mv_channels=out_dim,
            in_s_channels=in_dim * stride + stride,
            out_s_channels=out_dim,
        )

        # Normalization layer
        self.norm = EquiLayerNorm()

    def forward(self, node: Node) -> Node:
        if self.stride == 1:  # no pooling
            return Node(
                mv=node.mv,
                sc=node.sc,
                pos=node.pos,
                batch_idx=node.batch_idx,
                children=node,
            )

        with torch.no_grad():
            # Get batch indices from the first node in each group
            batch_idx = node.batch_idx[:: self.stride]

            # Calculate centers as mean of positions in each group
            centers = reduce(node.pos, "(n s) d -> n d", "mean", s=self.stride)

            # Calculate relative positions
            pos = rearrange(node.pos, "(n s) d -> n s d", s=self.stride)
            rel_pos = pos - centers[:, None]
            rel_distances = torch.norm(rel_pos, dim=-1)

        # Reshape multivectors and scalars
        mv = rearrange(node.mv, "(n s) c d -> n (s c) d", s=self.stride)
        sc = rearrange(node.sc, "(n s) c -> n (s c)", s=self.stride)

        # Add distance information to scalar features
        sc = torch.cat([sc, rel_distances.reshape(centers.shape[0], -1)], dim=-1)

        # Apply a single EquiLinear projection and normalization
        mv, sc = self.projection(mv, sc)
        mv, sc = self.norm(mv, sc)

        return Node(
            mv=mv,
            sc=sc,
            pos=centers,
            batch_idx=batch_idx,
            # tree_idx_rot=None,
            children=node,
        )


class BallUnpooling(nn.Module):
    """
    Ball unpooling (refinement) with equivariance:
        1. compute relative positions of children to the center of the ball
        2. use distances for scalar features to maintain rotation invariance
        3. apply equilinear projection and normalization
        4. output is a refined tree with the same number of nodes as before pooling
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride

        # Single EquiLinear for both multivectors and scalars
        self.projection = EquiLinear(
            in_mv_channels=in_dim,
            out_mv_channels=stride * out_dim,
            in_s_channels=in_dim + stride,  # Add space for distances
            out_s_channels=stride * out_dim,
        )

        # Normalization layer
        self.norm = EquiLayerNorm()

    def forward(self, node: Node) -> Node:
        with torch.no_grad():
            # Calculate relative positions of children to parent node
            rel_pos = (
                rearrange(node.children.pos, "(n m) d -> n m d", m=self.stride)
                - node.pos[:, None]
            )
            rel_distances = torch.norm(rel_pos, dim=-1)  # [n, stride]

        # Combine scalar features with distances
        sc = torch.cat([node.sc, rel_distances], dim=-1)

        # Process multivectors and scalars through the EquiLinear
        mv, sc = self.projection(node.mv, sc)

        # Reshape the projections to match the children's dimensions
        mv = rearrange(mv, "n (m d) e -> (n m) d e", m=self.stride)
        sc = rearrange(sc, "n (m d) -> (n m) d", m=self.stride)

        # Apply residual connection to children's features
        node.children.mv = node.children.mv + mv
        node.children.sc = node.children.sc + sc

        # Apply normalization
        node.children.mv, node.children.sc = self.norm(
            node.children.mv, node.children.sc
        )
        return node.children


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

        self.reference_mv = construct_reference_multivector(
            "canonical", torch.ones(dimensionality)
        )

    def forward(self, mv: torch.Tensor, sc: torch.Tensor, pos: torch.Tensor):

        # Store original for residual connection
        mv_residual, sc_residual = mv, sc

        # BallMSA.forward is (mv, sc, pos)
        mv, sc = self.BMSA(*self.norm1(mv, sc), pos)

        # First residual connection
        mv, sc = mv + mv_residual, sc + sc_residual

        # Store for second residual connection
        mv_residual, sc_residual = mv, sc

        # GeoMPL.forward is (mv, sc)
        mv, sc = self.geo_mlp(
            *self.norm2(mv, sc), reference_mv=self.reference_mv.to(mv.device)
        )

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
        dimensionality: int = 3,
        use_distance_bias: bool = False,
    ):
        super().__init__()
        hidden_dim = in_dim if direction == "down" else out_dim

        # No. of TransformerBlocks = depth
        self.blocks = nn.ModuleList(
            [
                ErwinTransformerBlock(
                    hidden_dim, num_heads, ball_size, mlp_ratio, dimensionality, use_distance_bias=use_distance_bias
                )
                for _ in range(depth)
            ]
        )
        self.rotate = [i % 2 for i in range(depth)] if rotate else [False] * depth

        self.pool = lambda node: node
        self.unpool = lambda node: node

        if direction == "down" and stride is not None:
            self.pool = BallPooling(hidden_dim, out_dim, stride)
        elif direction == "up" and stride is not None:
            self.unpool = BallUnpooling(in_dim, hidden_dim, stride, dimensionality)

    def forward(self, node: Node) -> Node:
        node = self.unpool(node)
        for blk in self.blocks:
            # Process the node through the block
            mv, sc = blk(node.mv, node.sc, node.pos)
            node.mv = mv
            node.sc = sc

        return self.pool(node)
        """
        # Simplified rotation check for clarity
        tree_idx_rot_inv = None
        if any(self.rotate):  # If any block in this layer might use rotation
            if node.tree_idx_rot is not None:
                tree_idx_rot_inv = torch.argsort(node.tree_idx_rot)
            # else: # Optional: Add an assertion or warning if rotation is expected but tree_idx_rot is None
            #     if any(r for r in self.rotate): # Only if some blocks actually rotate
            #         assert node.tree_idx_rot is not None, "tree_idx_rot must be provided for rotation if any block uses it"

        for i, blk in enumerate(self.blocks):
            current_block_rotates = self.rotate[i]
            # NOTE: !!! we haven't tested the if=true branch here, which would be the rotation case !!!
            if current_block_rotates:
                assert (
                    node.tree_idx_rot is not None
                ), "tree_idx_rot must be provided for rotation for this block"
                # Ensure tree_idx_rot_inv is computed if not already
                if (
                    tree_idx_rot_inv is None
                ):  # Should have been computed if any(self.rotate) and node.tree_idx_rot was not None
                    assert (
                        node.tree_idx_rot is not None
                    ), "Cannot rotate without tree_idx_rot"
                    tree_idx_rot_inv = torch.argsort(node.tree_idx_rot)

                mv_rotated = node.mv[node.tree_idx_rot]
                sc_rotated = node.sc[node.tree_idx_rot]
                pos_rotated = node.pos[node.tree_idx_rot]

                processed_mv_rotated, processed_sc_rotated = blk(
                    mv_rotated, sc_rotated, pos_rotated
                )

                node.mv = processed_mv_rotated[tree_idx_rot_inv]
                node.sc = processed_sc_rotated[tree_idx_rot_inv]
            else:
                processed_mv, processed_sc = blk(node.mv, node.sc, node.pos)
                node.mv = processed_mv
                node.sc = processed_sc
        return self.pool(node)
        """


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
        dimensionality (int): dimensionality of the input data.
        mp_steps (int): number of message passing steps in the MPNN Embedding.
        use_distance_bias (bool): whether to use distance-based attention bias.

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
        dimensionality: int = 3,
        mp_steps: int = 3,
        use_distance_bias: bool = False,
    ):
        super().__init__()
        assert len(enc_num_heads) == len(enc_depths) == len(ball_sizes)
        assert len(dec_num_heads) == len(dec_depths) == len(strides)
        assert len(strides) == len(ball_sizes) - 1

        self.rotate = rotate
        self.decode = decode
        self.ball_sizes = ball_sizes
        self.strides = strides

        self.embed = ErwinEmbedding(c_in, c_hidden[0], mp_steps, 16)

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
                    dimensionality=16,
                    use_distance_bias=use_distance_bias,
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
            dimensionality=16,
            use_distance_bias=use_distance_bias,
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
                        dimensionality=16,
                        use_distance_bias=use_distance_bias,
                    )
                )

        self.in_dim = c_in
        self.out_dim = c_hidden[0]
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
            node = layer(node)

        node = self.bottleneck(node)

        if self.decode:
            for layer in self.decoder:
                node = layer(node)
            return (
                node.mv[tree_mask][torch.argsort(tree_idx[tree_mask])],
                node.sc[tree_mask][torch.argsort(tree_idx[tree_mask])],
            )

        return node.mv, node.sc, node.batch_idx
