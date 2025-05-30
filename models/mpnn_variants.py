from __future__ import annotations

import torch
import torch.nn as nn
from gatr.layers import EquiLinear, EquiLayerNorm, GeoMLP
from gatr.layers.mlp import MLPConfig
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


class MPNN(nn.Module):
    """
    Message Passing Neural Network (see Gilmer et al., 2017).
        m_ij = MLP([h_i, h_j, pos_i - pos_j])       message
        m_i = mean(m_ij)                            aggregate
        h_i' = MLP([h_i, m_i])                      update
    """

    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()

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
        reference_mv: torch.Tensor,
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
        mv_messages, sc_messages = message_fn(mv_msg_input, sc_msg_input, reference_mv)

        # Aggregate messages per receiver node (col)
        mv_agg = scatter_mean(mv_messages, col, mv.size(0))
        sc_agg = scatter_mean(sc_messages, col, sc.size(0))

        # Update node features
        # mv_input_for_update: (N_nodes, 2 * N_channels, D_algebra)
        # sc_input_for_update: (N_nodes, 2 * N_channels_scalar)
        mv_update, sc_update = update_fn(
            torch.cat([mv, mv_agg], dim=1),
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
        reference_mv: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        edge_attr = self.compute_edge_attr(pos, edge_index)
        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            mv, sc = self.layer(message_fn, update_fn, mv, sc, reference_mv, edge_attr, edge_index)

        return mv, sc


class DistanceBasedScalarOnlyMPNN(nn.Module):
    """
    Distance-based Message Passing Neural Network that only processes scalar features.
    Uses pairwise distances instead of relative positions for equivariance.
        m_ij = Linear([h_i, h_j, dist_ij])          message
        m_i = mean(m_ij)                            aggregate
        h_i' = Linear([h_i, m_i])                   update
    """
    def __init__(self, dim: int, mp_steps: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()

        # Message functions: input is [h_i, h_j, dist_ij]
        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim + 1, mlp_ratio * dim),  # +1 for the scalar distance
                nn.GELU(),
                nn.Linear(mlp_ratio * dim, dim),
                nn.LayerNorm(dim)
            ) for _ in range(mp_steps)
        ])

        # Update functions: input is [h_i, m_i]
        self.update_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.LayerNorm(dim)
            ) for _ in range(mp_steps)
        ])

    def layer(
        self,
        message_fn: nn.Module,
        update_fn: nn.Module,
        h: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        row, col = edge_index

        # Construct message inputs by concatenating source and target features with distance
        msg_input = torch.cat([h[row], h[col], edge_attr], dim=-1)

        # Compute messages
        messages = message_fn(msg_input)

        # Aggregate messages per receiver node (col)
        message_agg = scatter_mean(messages, col, h.size(0))

        # Update node features with residual connection
        update = update_fn(torch.cat([h, message_agg], dim=-1))
        return h + update

    @torch.no_grad()
    def compute_edge_attr(self, pos, edge_index):
        # Compute pairwise distances (scalar) instead of relative positions (vector)
        # Returns shape (num_edges, 1) for concatenation with node features
        return torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1, keepdim=True)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        edge_attr = self.compute_edge_attr(pos, edge_index)
        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            x = self.layer(message_fn, update_fn, x, edge_attr, edge_index)

        return x
