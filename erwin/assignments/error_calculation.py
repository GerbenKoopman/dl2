import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from invariance_error import invariance_error_new, invariance_error
from balltree import build_balltree_with_rotations
from models.erwin import ErwinTransformer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_csv(file: Path, title: str, x_label: str, y_label: str):
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Group by run_name and calculate mean and std
    stats = df.groupby('run_name').agg({
        'error': ['mean', 'std']
    }).reset_index()
    
    stats.columns = ['run_name', 'mean', 'std']
    
    # Define the specific order of configurations
    order = [
        "original_dynamic_tree",
        "original_fixed_tree",
        "fixed_eq9",
        "fixed_eq9_eq12",
        "fixed_eq9_eq12_eq13"
    ]
    
    # Reorder the DataFrame based on the specified order
    stats['order'] = stats['run_name'].map({name: i for i, name in enumerate(order)})
    stats = stats.sort_values('order').reset_index(drop=True)
    
    # Create better display names for the x-axis
    display_names = [
        "Original\nDynamic Tree",
        "Original\nFixed Tree",
        "Fixed Eq9\nFixed Tree",
        "Fixed Eq9+12\nFixed Tree",
        "Fixed Eq9+12+13\nFixed Tree"
    ]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot bar chart with error bars
    bars = plt.bar(display_names, stats['mean'], yerr=stats['std'], capsize=5)
    
    # Add numerical values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + stats['std'][i] * 1.1,
                 f'{stats["mean"][i]:.2e}',  # Scientific notation
                 ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=0)  # Changed to 0 since we're using newlines in labels
    plt.tight_layout()
    plt.yscale("log")
    
    # Save the plot
    plt.savefig(file.with_suffix(".png"))
    plt.close()

def calculate_error(config_name: str,
                    model_config: dict,
                    run_name: str = "fixed_tree",
                    batch_size: int = 1,
                    num_points: int = 1024,
                    theta: float = np.pi / 4.5,
                    fixed_balltree: bool = False,
                    seed: int = None,
                    use_new_error: bool = True,
                    output_suffix: str = ""):
    
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
        
    print(f"Calculating error for {config_name} with run name {run_name}, seed {seed}")
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

    # Choose which error function to use
    error_fn = invariance_error_new if use_new_error else invariance_error
    
    out = error_fn(
        model,
        node_features,
        node_positions,
        rotation_matrix,
        batch_idx,
        **kwargs,
    )

    # Append out to .csv file
    out_dir = Path(__file__).parent.resolve() / "invariance_error"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construct file name with suffix
    csv_filename = f"{config_name}{output_suffix}.csv"
    
    # Append to the CSV file
    with open(out_dir / csv_filename, "a") as f:
        f.write(f"{run_name},{out.item()},{seed}\n")

    print(f"Invariance error: {out}")
    return out.item()

def main():
    parser = argparse.ArgumentParser(description='Run invariance error tests on Erwin model.')
    parser.add_argument('--use_new_error', action='store_true', 
                        help='Use the new invariance_error_new function instead of the original')
    parser.add_argument('--suffix', type=str, default=None, 
                        help='Custom suffix to add to output files (CSV and plots). If not provided, defaults to "_new" or "_old" based on error function')
    parser.add_argument('--num_runs', type=int, default=5, 
                        help='Number of runs per configuration')
    parser.add_argument('--base_seed', type=int, default=42, 
                        help='Base seed for random number generation')
    args = parser.parse_args()
    
    # Generate seeds based on base seed
    seeds = [args.base_seed + i * 100 for i in range(args.num_runs)]
    
    # Set default suffix based on error function if not provided
    if args.suffix is None:
        args.suffix = '_new' if args.use_new_error else '_old'
    # Add underscore to suffix if it's not empty and doesn't start with one
    elif args.suffix and not args.suffix.startswith('_'):
        args.suffix = '_' + args.suffix
    
    # Set a fixed seed for reproducibility
    set_seed(args.base_seed)
    
    # Get current path
    pwd = Path(__file__).parent.resolve()
    
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
        "dec_num_heads": [1],  # Added manually - NOTE: the other group set it [4,] according to Maksim: https://canvas.uva.nl/groups/389353/discussion_topics/969271
        "dec_depths": [1],  # Added manually - NOTE: the other group set it [4,] according to Maksim: https://canvas.uva.nl/groups/389353/discussion_topics/969271
        "strides": [4],  # 0.25 coarsening
        "mp_steps": 0,  # no MPNN
        "decode": True,  # no decoder
        "dimensionality": 2,  # for visualization
        "rotate": 0,
    }

    # Define different configurations to test
    test_configs = [
        {"name": "original_dynamic_tree", "fixed_balltree": False, "fix_eq9": False, "fix_eq12": False, "fix_eq13": False},
        {"name": "original_fixed_tree", "fixed_balltree": True, "fix_eq9": False, "fix_eq12": False, "fix_eq13": False},
        {"name": "fixed_eq9", "fixed_balltree": True, "fix_eq9": True, "fix_eq12": False, "fix_eq13": False},
        {"name": "fixed_eq9_eq12", "fixed_balltree": True, "fix_eq9": True, "fix_eq12": True, "fix_eq13": False},
        {"name": "fixed_eq9_eq12_eq13", "fixed_balltree": True, "fix_eq9": True, "fix_eq12": True, "fix_eq13": True},
    ]

    for name, model_config in [("single", config_single), ("pool", config_pool)]:
        # Clear previous CSV file
        out_dir = pwd / "invariance_error"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_file = out_dir / f"{name}{args.suffix}.csv"
        
        # Create a new CSV file with header
        with open(csv_file, "w") as f:
            f.write("run_name,error,seed\n")
        
        for test in test_configs:
            # Add fix parameters to model config
            model_config_with_fixes = model_config.copy()
            model_config_with_fixes["fix_eq9"] = test["fix_eq9"]
            model_config_with_fixes["fix_eq12"] = test["fix_eq12"]
            model_config_with_fixes["fix_eq13"] = test["fix_eq13"]
            
            # Run multiple times with different seeds
            for seed_idx in range(args.num_runs):
                seed = seeds[seed_idx]
                # Calculate the error
                calculate_error(
                    name, 
                    model_config_with_fixes, 
                    run_name=test["name"], 
                    fixed_balltree=test["fixed_balltree"],
                    seed=seed,
                    use_new_error=args.use_new_error,
                    output_suffix=args.suffix
                )
        
        # Generate plot for this model configuration
        error_type = "new" if args.use_new_error else "original"
        plot_csv(
            csv_file,
            title=f"Invariance error ({name} model, {error_type} error function)",
            x_label="Configuration",
            y_label="Invariance error",
        )

if __name__ == "__main__":
    main()
