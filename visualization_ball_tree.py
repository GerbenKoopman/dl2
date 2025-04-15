import numpy as np
from balltree import build_balltree
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plots

# Parameters
bs, num_points, dim = 2, 64, 3
device = "cpu"

# Generate random points and batch indices
points = torch.rand(num_points * bs, dim, dtype=torch.float32, device=device)
batch_idx = torch.repeat_interleave(torch.arange(bs, device=device), num_points)

# Build ball tree
tree_idx, tree_mask = build_balltree(points, batch_idx)
grouped_points = points[tree_idx]

# Node size function
level_to_node_size = lambda level: 2**level

# Plot each level
for level in range(0, 6):
    node_size = level_to_node_size(level)
    if grouped_points.shape[0] % node_size != 0:
        print(f"Skipping level {level} due to shape mismatch.")
        continue

    groups = grouped_points.reshape(-1, node_size, dim)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Different color per group
    colors = plt.get_cmap("tab20", groups.shape[0])

    for i, group in enumerate(groups):
        group_np = group.cpu().numpy()
        ax.scatter(
            group_np[:, 0],
            group_np[:, 1],
            group_np[:, 2],
            color=colors(i),
            label=f"Group {i}",
            s=10,
        )

    ax.set_title(f"Ball Tree Level {level} - {groups.shape[0]} Groups")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=30)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"level_{level}.png")
    plt.close()

print("Saved ball tree visualizations for levels 0 through 5.")
