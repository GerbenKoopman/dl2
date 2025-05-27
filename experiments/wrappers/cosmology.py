import torch
import torch.nn as nn
from gatr.layers import EquiLinear, ScalarGatedNonlinearity
from gatr.interface import (
    embed_point,
    embed_scalar,
    embed_translation,
    embed_oriented_plane,
    extract_point,
)


class Embedding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.pos_embedding = EquiLinear(1, out_dim, 1, out_dim)

    def forward(self, pos):
        return self.pos_embedding(embed_point(pos).unsqueeze(1))


class CosmologyModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model
        self.embedding_model = Embedding(main_model.in_dim)

        # First EquiLinear layer
        self.pred_head1 = EquiLinear(
            in_mv_channels=main_model.out_dim,
            out_mv_channels=main_model.out_dim,
            in_s_channels=main_model.out_dim,
            out_s_channels=main_model.out_dim,
        )
        # Nonlinearity
        self.nonlin = ScalarGatedNonlinearity("gelu")
        # Second EquiLinear layer
        self.pred_head2 = EquiLinear(
            in_mv_channels=main_model.out_dim,
            out_mv_channels=1,
            in_s_channels=main_model.out_dim,
            out_s_channels=1,
        )

    def forward(self, node_positions, **kwargs):
        node_features_mv, node_features_sc = self.embedding_model(node_positions)

        # Run the main model
        mv_output, sc_output = self.main_model(
            node_features_mv, node_features_sc, node_positions, **kwargs
        )

        # First layer
        mv_hidden, sc_hidden = self.pred_head1(mv_output, sc_output)
        # Nonlinearity
        mv_hidden, sc_hidden = self.nonlin(mv_hidden, sc_hidden)
        # Second layer
        mv_pred, sc_pred = self.pred_head2(mv_hidden, sc_hidden)

        # Extract translation components (bivector indices 4, 5, 6)
        # velocity = mv_pred[..., [4, 5, 6]]
        velocity = extract_point(mv_pred, divide_by_embedding_dim=True)

        # Reshape to remove the channel dimension
        velocity = velocity.squeeze(1)  # Result: [bs*nodes, 3]

        return velocity

    def step(self, batch, prefix="train"):
        pred = self(batch["pos"], **batch)
        loss = ((pred - batch["target"]) ** 2).mean()
        return {f"{prefix}/loss": loss}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
