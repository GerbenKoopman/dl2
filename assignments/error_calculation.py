import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from invariance_error import invariance_error
from balltree import build_balltree_with_rotations
from models.erwin import ErwinTransformer

# Get the current path
pwd = Path(__file__).parent.resolve()


def calculate_error(config_name: str,
                    model_config: dict,
                    run_name: str = "fixed_tree",
                    batch_size: int = 1,
                    num_points: int = 1024,
                    theta: float = np.pi / 4.5,
                    fixed_balltree: bool = False):
    print(f"Calculating error for {config_name} with run name {run_name}")
    model = ErwinTransformer(**model_config).cuda()

    # Calculate the points, features, batch indeces, and the ball tree
    node_features = torch.randn(num_points * batch_size, model_config["c_in"]).cuda()
    node_positions = torch.rand(num_points * batch_size, model_config["dimensionality"]).cuda()
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_points).cuda()

    rotation_matrix = torch.tensor(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=torch.float32,
    ).cuda()

    kwargs = {}
    if fixed_balltree:
        tree_idx, tree_mask, tree_idx_rot = build_balltree_with_rotations(
            node_positions,
            batch_idx,
            model_config["strides"],
            model_config["ball_sizes"],
            model_config["rotate"],
        )
        kwargs["tree_idx"] = tree_idx
        kwargs["tree_mask"] = tree_mask
        kwargs["tree_idx_rot"] = tree_idx_rot

    out = invariance_error(
        model,
        node_features,
        node_positions,
        rotation_matrix,
        batch_idx,
        **kwargs,
    )

    # Append out to .csv file
    out_dir = pwd / "invariance_error"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{config_name}.csv", "a") as f:
        f.write(f"{run_name},{out.item()}\n")

    print(f"Invariance error: {out}")


config_single = {
    "c_in": 16,
    "c_hidden": 16,
    "ball_sizes": [128],
    "enc_num_heads": [1],
    "enc_depths": [1],
    "dec_num_heads": [],
    "dec_depths": [],
    "strides": [],  # no coarsening
    "mp_steps": 0,  # no MPNN
    "decode": True,  # no decoder
    "dimensionality": 2,  # for visualization
    "rotate": 0,
}
config_pool = {
    "c_in": 16,
    "c_hidden": 16,
    "ball_sizes": [128, 128],
    "enc_num_heads": [1, 1],
    "enc_depths": [1, 1],
    "dec_num_heads": [1],  # Added manually
    "dec_depths": [1],  # Added manually
    "strides": [4],  # 0.25 coarsening
    "mp_steps": 0,  # no MPNN
    "decode": True,  # no decoder
    "dimensionality": 2,  # for visualization
    "rotate": 0,
}


for name, model_config in [("single", config_single), ("pool", config_pool)]:
    # Calculate the error with and without fixed balltree
    calculate_error(name, model_config, "fixed_tree-eq9-eq12", fixed_balltree=True)
