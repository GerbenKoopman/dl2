# Implement the invariance test as a function
# ’invariance error(model, x: Tensor, pos: Tensor, R: Tensor, **kwargs) → float’
# that computes the invariance error of ERWIN for a batch of data.
# ||Erwin(x, pos) − Erwin(x, pos @ R)||

import sys
import os

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.erwin import ErwinTransformer


def invariance_error(
    model,
    x: torch.Tensor,
    pos: torch.Tensor,
    R: torch.Tensor,
    batch_idx: torch.Tensor,
    **kwargs,
) -> float:
    """
    Computes the invariance error of ERWIN for a batch of data.

    Parameters:
    - model: The ERWIN model.
    - x: Feature tensor of shape (N, D), where N is the number of points and D is the dimensionality.
    - pos: Positions tensor of shape (N, D).
    - R: Rotation matrix of shape (D, D).
    - batch_idx: Batch indices tensor of shape (N,).
    - **kwargs: Additional arguments to pass to the model.

    Returns:
    - float: The invariance error.
    """

    # Rotate the input points
    pos_rotated = torch.matmul(pos, R.T)

    # Compute the output for both original and rotated inputs
    output_original = model(x, pos, batch_idx, **kwargs)
    output_rotated = model(x, pos_rotated, batch_idx, **kwargs)

    # Compute the invariance error
    error = torch.norm(output_original - output_rotated, p=2)

    return error


config = {
    "c_in": 16,
    "c_hidden": 16,
    "ball_sizes": [128],
    "enc_num_heads": [
        1,
    ],
    "enc_depths": [
        1,
    ],
    "dec_num_heads": [],
    "dec_depths": [],
    "strides": [],
    "mp_steps": 0,
    "decode": True,
    "dimensionality": 2,
    "rotate": 0,
}
model = ErwinTransformer(**config).cuda()
bs = 1
num_points = 1024
node_features = torch.randn(num_points * bs, config["c_in"]).cuda()
node_positions = torch.rand(num_points * bs, config["dimensionality"]).cuda()
batch_idx = torch.repeat_interleave(torch.arange(bs), num_points).cuda()

theta = np.pi / 4  # 45-degree rotation
rotation_matrix = torch.tensor(
    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
    dtype=torch.float32,
).cuda()

out = invariance_error(
    model,
    node_features,
    node_positions,
    rotation_matrix,
    batch_idx,
)

print(f"Invariance error: {out}")
