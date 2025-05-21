import torch
import torch.nn as nn
from gatr.layers import EquiLinear
from gatr.interface import (
    embed_point,
    embed_scalar,
    embed_translation,
    embed_oriented_plane,
)

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos):
        mv = embed_point(pos)

        return mv.unsqueeze(-2)


class CosmologyModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model
        self.embedding_model = Embedding()
        self.pred_head = EquiLinear(
            in_mv_channels=main_model.out_dim,
            out_mv_channels=1,
            in_s_channels=main_model.out_dim,
            out_s_channels=1
        )

    def forward(self, node_positions, **kwargs):
        node_features_mv = self.embedding_model(node_positions)  # Shape [bs*nodes, 16]
        
        # Create scalar features with 16 channels as all zeros: WHY ALL ZEROS? => maybe worth to try all ones etc.
        node_features_sc = torch.zeros(node_features_mv.shape[0], 16, device=node_features_mv.device)
        
        # Run the main model
        mv_output, sc_output = self.main_model(node_features_mv, node_features_sc, node_positions, **kwargs)
        
        # Process through EquiLinear prediction head
        mv_pred, sc_pred = self.pred_head(mv_output, sc_output)
        
        # Extract translation components (bivector indices 4, 5, 6)
        velocity = mv_pred[..., [4, 5, 6]]
        
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
