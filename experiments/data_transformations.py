from scipy.spatial.transform import Rotation
import numpy as np
import torch
import copy
from torch.utils.data import Dataset


class RotatedCosmologyDataset(Dataset):
    """Wrapper around CosmologyDataset that applies random transformation."""

    def __init__(self, base_dataset, device="cpu", seed=0):
        self.base_dataset = base_dataset
        self.device = device
        self.seed = seed

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Get random rotation matrix using Euler angles
        self.rotation = Rotation.random()
        self.angles = self.rotation.as_euler("xyz")
        self.rotation = torch.tensor(
            self.rotation.as_matrix(), dtype=torch.float32, device=self.device
        )

        # Randomly apply reflection
        # Get determinant to decide to flip handedness or not
        self.determinant = np.random.choice([-1, 1])

        if self.determinant == -1:
            # Simple way: flip one coordinate axis to change det(+1) -> det(-1)
            self.rotation[0, :] = -self.rotation[
                0, :
            ]  # Flip first row (equivalent to reflecting through yz-plane)

    def __len__(self):
        return len(self.base_dataset)

    @property
    def collate_fn(self):
        """Delegate collate_fn to the base dataset for DataLoader compatibility."""
        return self.base_dataset.collate_fn

    def __getitem__(self, idx):
        data = copy.deepcopy(self.base_dataset[idx])

        # Apply transformation to positions and velocities
        positions = data["pos"]
        velocities = data["target"]

        # Ensure tensors are on the same device as rotation matrix
        positions = positions.to(self.device)
        velocities = velocities.to(self.device)

        transformed_positions = torch.matmul(positions, self.rotation.T)
        transformed_velocities = torch.matmul(velocities, self.rotation.T)

        # Update data with transformed values
        data["pos"] = transformed_positions
        data["target"] = transformed_velocities

        return data
