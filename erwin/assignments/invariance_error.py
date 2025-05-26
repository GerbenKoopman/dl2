import torch
import copy


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
    output_original = model(x, pos, batch_idx, **copy.deepcopy(kwargs))
    output_rotated = model(x, pos_rotated, batch_idx, **copy.deepcopy(kwargs))

    # Take the mean of the outputs over the batch dimension
    output_original = output_original.mean(dim=0)
    output_rotated = output_rotated.mean(dim=0)

    # Compute the invariance error as the normalized ratio of the difference
    normalized_ratio = torch.norm(output_original - output_rotated) / torch.norm(
        output_rotated
    )

    return normalized_ratio


def invariance_error_new(
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
    output_original = model(x, pos, batch_idx, **copy.deepcopy(kwargs))
    output_rotated = model(x, pos_rotated, batch_idx, **copy.deepcopy(kwargs))

    # Calculate point-wise errors, normalizing by original
    point_errors = torch.norm(output_original - output_rotated, dim=1) / torch.norm(output_original, dim=1)
    
    # Take mean at the end over all points
    return point_errors.mean()