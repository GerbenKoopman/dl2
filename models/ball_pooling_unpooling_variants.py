from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, reduce
from dataclasses import dataclass

from gatr.layers import EquiLinear, EquiLayerNorm
from gatr.interface import embed_translation

# Node dataclass (copied from erwin.py for self-containment of this file)
@dataclass
class Node:
    """Dataclass to store the hierarchical node information."""

    mv: torch.Tensor
    sc: torch.Tensor
    pos: torch.Tensor
    batch_idx: torch.Tensor
    tree_idx_rot: torch.Tensor | None = None
    children: Node | None = None


class BallPoolingRelDist(nn.Module):
    """
    Pooling: Concatenates leaf nodes. Relative distances to ball center are added to scalar features.
    """

    def __init__(
        self, in_dim: int, out_dim: int, stride: int = 2, dimensionality: int = 3 # dimensionality is not used here but kept for signature consistency
    ):
        super().__init__()
        self.stride = stride

        self.projection = EquiLinear(
            in_mv_channels=in_dim * stride, # Only original mv channels
            out_mv_channels=out_dim,
            in_s_channels=in_dim * stride + stride, # Original scalar channels + stride for distances
            out_s_channels=out_dim,
        )
        self.norm = EquiLayerNorm()

    def forward(self, node: Node) -> Node:
        if self.stride == 1:
            return Node(
                mv=node.mv,
                sc=node.sc,
                pos=node.pos,
                batch_idx=node.batch_idx,
                children=node,
            )

        with torch.no_grad():
            batch_idx = node.batch_idx[:: self.stride]
            centers = reduce(node.pos, "(n s) d -> n d", "mean", s=self.stride)
            pos_reshaped = rearrange(node.pos, "(n s) d -> n s d", s=self.stride)
            rel_pos = pos_reshaped - centers[:, None]
            rel_distances = torch.norm(rel_pos, dim=-1)

        mv = rearrange(node.mv, "(n s) c d -> n (s c) d", s=self.stride)
        sc = rearrange(node.sc, "(n s) c -> n (s c)", s=self.stride)
        sc = torch.cat([sc, rel_distances.reshape(centers.shape[0], -1)], dim=-1)

        mv, sc = self.projection(mv, sc)
        mv, sc = self.norm(mv, sc)

        return Node(
            mv=mv,
            sc=sc,
            pos=centers,
            batch_idx=batch_idx,
            children=node,
        )


class BallUnpoolingRelDist(nn.Module):
    """
    Unpooling: Relative distances of children to parent are added to scalar features.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3  # dimensionality is not used here but kept for signature consistency
    ):
        super().__init__()
        self.stride = stride

        self.projection = EquiLinear(
            in_mv_channels=in_dim, # Only original mv channels
            out_mv_channels=stride * out_dim,
            in_s_channels=in_dim + stride, # Original scalar channels + stride for distances
            out_s_channels=stride * out_dim,
        )
        self.norm = EquiLayerNorm()

    def forward(self, node: Node) -> Node:
        with torch.no_grad():
            rel_pos = (
                rearrange(node.children.pos, "(n m) d -> n m d", m=self.stride)
                - node.pos[:, None]
            )
            rel_distances = torch.norm(rel_pos, dim=-1)

        sc = torch.cat([node.sc, rel_distances], dim=-1)
        mv_orig = node.mv # Use original mv directly

        mv, sc = self.projection(mv_orig, sc) # Pass original mv

        mv = rearrange(mv, "n (m d) e -> (n m) d e", m=self.stride)
        sc = rearrange(sc, "n (m d) -> (n m) d", m=self.stride)

        node.children.mv = node.children.mv + mv
        node.children.sc = node.children.sc + sc
        node.children.mv, node.children.sc = self.norm(
            node.children.mv, node.children.sc
        )
        return node.children


class BallPoolingRelDistRelPosMv(nn.Module):
    """
    Pooling: Concatenates leaf nodes.
    - Relative distances to ball center are added to scalar features.
    - Embedded relative positions (translations) are added to multivector features.
    """

    def __init__(
        self, in_dim: int, out_dim: int, stride: int = 2, dimensionality: int = 3
    ):
        super().__init__()
        self.stride = stride
        self.dimensionality = dimensionality

        self.projection = EquiLinear(
            in_mv_channels=in_dim * stride + stride,  # Original mv channels + stride for embedded translations
            out_mv_channels=out_dim,
            in_s_channels=in_dim * stride + stride, # Original scalar channels + stride for distances
            out_s_channels=out_dim,
        )
        self.norm = EquiLayerNorm()

    def forward(self, node: Node) -> Node:
        if self.stride == 1:
            return Node(
                mv=node.mv,
                sc=node.sc,
                pos=node.pos,
                batch_idx=node.batch_idx,
                children=node,
            )

        with torch.no_grad():
            batch_idx = node.batch_idx[:: self.stride]
            centers = reduce(node.pos, "(n s) d -> n d", "mean", s=self.stride)
            pos_reshaped = rearrange(node.pos, "(n s) d -> n s d", s=self.stride)
            rel_pos = pos_reshaped - centers[:, None]
            rel_distances = torch.norm(rel_pos, dim=-1)
            embedded_rel_pos_mv = embed_translation(rel_pos)

        mv_orig_reshaped = rearrange(node.mv, "(n s) c d -> n (s c) d", s=self.stride)
        mv = torch.cat([mv_orig_reshaped, embedded_rel_pos_mv], dim=1)

        sc = rearrange(node.sc, "(n s) c -> n (s c)", s=self.stride)
        sc = torch.cat([sc, rel_distances.reshape(centers.shape[0], -1)], dim=-1)

        mv, sc = self.projection(mv, sc)
        mv, sc = self.norm(mv, sc)

        return Node(
            mv=mv,
            sc=sc,
            pos=centers,
            batch_idx=batch_idx,
            children=node,
        )


class BallUnpoolingRelDistRelPosMv(nn.Module):
    """
    Unpooling:
    - Relative distances of children to parent are added to scalar features.
    - Embedded relative positions (translations) of children to parent are added to multivector features.
    """
    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.dimensionality = dimensionality

        self.projection = EquiLinear(
            in_mv_channels=in_dim + stride,  # Original mv channels + stride for embedded translations
            out_mv_channels=stride * out_dim,
            in_s_channels=in_dim + stride,  # Original scalar channels + stride for distances
            out_s_channels=stride * out_dim,
        )
        self.norm = EquiLayerNorm()

    def forward(self, node: Node) -> Node:
        with torch.no_grad():
            rel_pos = (
                rearrange(node.children.pos, "(n m) d -> n m d", m=self.stride)
                - node.pos[:, None]
            )
            rel_distances = torch.norm(rel_pos, dim=-1)
            embedded_rel_pos_mv = embed_translation(rel_pos)

        sc = torch.cat([node.sc, rel_distances], dim=-1)
        mv_combined = torch.cat([node.mv, embedded_rel_pos_mv], dim=1)

        mv, sc = self.projection(mv_combined, sc)

        mv = rearrange(mv, "n (m d) e -> (n m) d e", m=self.stride)
        sc = rearrange(sc, "n (m d) -> (n m) d", m=self.stride)

        node.children.mv = node.children.mv + mv
        node.children.sc = node.children.sc + sc
        node.children.mv, node.children.sc = self.norm(
            node.children.mv, node.children.sc
        )
        return node.children 