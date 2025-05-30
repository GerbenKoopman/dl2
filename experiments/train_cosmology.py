import sys

sys.path.append("../../")

import argparse
import torch

torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from erwin.training import fit, to_cuda, validate
from erwin.models.erwin import ErwinTransformer
from erwin.experiments.datasets import CosmologyDataset
from erwin.experiments.wrappers import CosmologyModel

from data_transformations import RotatedCosmologyDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="erwin",
        help="Model type (mpnn, pointtransformer, erwin)",
    )
    parser.add_argument("--data-path", type=str)
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=[
            "custom_geo",
            "smallest",
            "smallest_mp_original",
            "smallest_mp_scalar",
            "small",
            "medium",
            "large",
        ],
        help="Model size configuration",
    )
    parser.add_argument(
        "--num-samples", type=int, default=8192, help="Number of samples for training"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3000, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Whether to use Weights & Biases for logging",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--val-every-iter",
        type=int,
        default=500,
        help="Validation frequency in iterations",
    )
    parser.add_argument(
        "--experiment", type=str, default="glx_node", help="Experiment name"
    )
    parser.add_argument(
        "--test", action="store_true", default=True, help="Whether to run testing"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="RelDistRelPosMv",
        choices=["RelDist", "RelDistRelPosMv"],
        help="Type of pooling strategy",
    )
    parser.add_argument(
        "--unpooling-type",
        type=str,
        default="RelDistRelPosMv",
        choices=["RelDist", "RelDistRelPosMv"],
        help="Type of unpooling strategy",
    )
    parser.add_argument(
        "--use-distance-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use distance-based attention bias in BallMSA (overrides config and model default)",
    )
    parser.add_argument(
        "--dimensionality",
        type=int,
        default=3,
        help="Spatial dimensionality of the input data (e.g., 3 for 3D points)",
    )
    parser.add_argument(
        "--algebra-dimensionality",
        type=int,
        default=16,
        help="Dimensionality of the geometric algebra (e.g., 16 for G(3,0,1))",
    )
    parser.add_argument(
        "--mp-steps",
        type=int,
        default=3,  # Default to 0, specific configs can override, CLI can override further
        help="Number of message passing steps in the MPNN Embedding",
    )
    parser.add_argument(
        "--mpnn-type",
        type=str,
        default="original",
        choices=[
            "scalar_only",
            "original",
        ],  # scalar only is faster but less expressive
        help="Type of MPNN to use",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for the model (default: 0.0)",
    )

    return parser.parse_args()


erwin_configs = {
    # Simplified configs: pooling, unpooling, dim, algebra_dim, use_dist_bias removed
    # mp_steps and mpnn_type remain for specific variants, but can be overridden by CLI
    "custom_geo": {
        "c_in": 4,
        "c_hidden": [4, 8],
        "enc_num_heads": [2, 4],
        "enc_depths": [2, 2],
        "dec_num_heads": [2],
        "dec_depths": [2],
        "strides": [2],
        "ball_sizes": [128, 128],
        "rotate": 0,
        "mp_steps": 3,  # Explicitly 3 for non-MPNN version
        "mpnn_type": "original",  # Default, relevant if mp_steps > 0
        "dropout": 0.0,
    },
    "smallest": {
        "c_in": 8,
        "c_hidden": [8, 16],
        "enc_num_heads": [2, 4],
        "enc_depths": [2, 2],
        "dec_num_heads": [2],
        "dec_depths": [2],
        "strides": [2],
        "ball_sizes": [128, 128],
        "rotate": 0,
        "mp_steps": 0,  # Explicitly 0 for non-MPNN version
        "mpnn_type": "original",  # Default, relevant if mp_steps > 0
        "dropout": 0.0,
    },
    "small": {
        "c_in": 32,
        "c_hidden": [32, 64, 128, 256],
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "rotate": 0,
        "ball_sizes": [256, 256, 256, 256],
        "mp_steps": 0,  # Will be overridden by CLI
        "mpnn_type": "original",  # Will be overridden by CLI
        "dropout": 0.0,
    },
    "medium": {
        "c_in": 64,
        "c_hidden": [64, 128, 256, 512],
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "rotate": 0,
        "ball_sizes": [512, 512, 512, 512],
        "mp_steps": 0,  # Will be overridden by CLI
        "mpnn_type": "original",  # Will be overridden by CLI
        "dropout": 0.0,
    },
    "large": {
        "c_in": 128,
        "c_hidden": [128, 256, 512, 1024],
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "rotate": 0,
        "ball_sizes": [256, 256, 256, 256],
        "mp_steps": 0,  # Will be overridden by CLI
        "mpnn_type": "original",  # Will be overridden by CLI
        "dropout": 0.0,
    },
}

model_cls = {
    "erwin": ErwinTransformer,
}


def evaluate_robustness(model, test_dataset, config, num_transforms=3):
    """Evaluate model robustness on transformed data."""

    # Original performance
    original_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=4,
    )

    original_stats = validate(model, original_loader, config)
    original_loss = original_stats["avg/val/loss"]
    print(f"Original loss: {original_loss:.4f}")

    # Test on transformed data
    transform_results = []
    for i in range(num_transforms):
        transformed_dataset = RotatedCosmologyDataset(
            test_dataset, device="cpu", seed=i
        )
        transformed_loader = DataLoader(
            transformed_dataset,
            batch_size=config.get("batch_size", 16),
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
            num_workers=4,
        )

        transformed_stats = validate(model, transformed_loader, config)
        transformed_loss = transformed_stats["avg/val/loss"]

        # Get transformation parameters
        angles = transformed_dataset.angles
        determinant = transformed_dataset.determinant

        transform_result = {
            "loss": transformed_loss,
            "angles": angles,
            "determinant": determinant,
        }
        transform_results.append(transform_result)

        print(
            f"[{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}], {determinant}, {transformed_loss:.4f}"
        )

    return {"original_loss": original_loss, "transform_results": transform_results}


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.model == "erwin":
        # Start with the base config for the chosen size
        model_config = erwin_configs[args.size].copy()

        # Override/set parameters directly from CLI arguments
        # These CLI arguments will always take precedence
        model_config["pooling_type"] = args.pooling_type
        model_config["unpooling_type"] = args.unpooling_type
        model_config["dimensionality"] = args.dimensionality
        model_config["algebra_dimensionality"] = args.algebra_dimensionality
        model_config["mpnn_type"] = args.mpnn_type or "original"
        model_config["mp_steps"] = args.mp_steps or 0
        model_config["dropout"] = args.dropout or 0.0

        # Handle use_distance_bias:
        if args.use_distance_bias is not None:
            model_config["use_distance_bias"] = args.use_distance_bias

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    train_dataset = CosmologyDataset(
        task="node",
        split="train",
        num_samples=args.num_samples,
        tfrecords_path=args.data_path,
        knn=10,
    )
    val_dataset = CosmologyDataset(
        task="node",
        split="val",
        num_samples=512,
        tfrecords_path=args.data_path,
        knn=10,
    )
    test_dataset = CosmologyDataset(
        task="node",
        split="test",
        num_samples=512,
        tfrecords_path=args.data_path,
        knn=10,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )

    dynamic_model = model_cls[args.model](**model_config)
    model = CosmologyModel(dynamic_model)
    model = to_cuda(model)
    # model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    config = vars(args)
    config.update(model_config)

    # Run training
    fit(
        config,
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        test_loader,
        100,
        200,
    )

    # After training, evaluate robustness
    if args.test:
        print("\n" + "=" * 50)
        print("ROBUSTNESS EVALUATION")
        print("=" * 50)
        robustness_results = evaluate_robustness(model, test_dataset, config)
        print("\nRobustness Results dict:")
        print(robustness_results)
