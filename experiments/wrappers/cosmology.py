import torch
import torch.nn as nn

from gatr.interface import (
    embed_point,
    embed_scalar,
    embed_translation,
    embed_oriented_plane,
)


# masses = inputs[:, :, [0]]  # (batchsize, objects, 1)
# points = inputs[:, :, 1:4]  # (batchsize, objects, 3)
# velocities = inputs[:, :, 4:7]  # (batchsize, objects, 3)


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos):
        mv = embed_point(pos)

        return mv.unsqueeze(2)


class CosmologyModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model
        self.embedding_model = Embedding()
        self.pred_head = nn.Sequential(
            nn.Linear(main_model.out_dim, main_model.out_dim),
            nn.GELU(),
            nn.Linear(main_model.out_dim, 3),
        )

    def forward(self, node_positions, **kwargs):
        node_features = self.embedding_model(node_positions)
        return self.pred_head(self.main_model(node_features, node_positions, **kwargs))

    def step(self, batch, prefix="train"):
        pred = self(batch["pos"], **batch)
        loss = ((pred - batch["target"]) ** 2).mean()
        return {f"{prefix}/loss": loss}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
