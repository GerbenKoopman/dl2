# Original MPNN with RelDist only pooling/unpooling (distance-based)
python train_cosmology.py --data-path ../datasets/cosmology --size smallest --batch-size 2 --use-wandb --experiment "original_mpnn_reldist_pool" --num-samples 8192 --pooling-type RelDist --unpooling-type RelDist --mpnn-type original --mp-steps 3

# Original MPNN with RelDistRelPosMv pooling/unpooling (using both distance and embedded relative positions)
python train_cosmology.py --data-path ../datasets/cosmology --size smallest --batch-size 2 --use-wandb --experiment "original_mpnn_reldistrelposmv_pool" --num-samples 8192 --pooling-type RelDistRelPosMv --unpooling-type RelDistRelPosMv --mpnn-type original --mp-steps 3

# Scalar-only MPNN with RelDist pooling/unpooling (distance-based)
python train_cosmology.py --data-path ../datasets/cosmology --size smallest --batch-size 2 --use-wandb --experiment "scalar_mpnn_reldist_pool" --num-samples 8192 --pooling-type RelDist --unpooling-type RelDist --mpnn-type scalar_only --mp-steps 3

# Scalar-only MPNN with RelDistRelPosMV pooling/unpooling (using both distance and embedded relative positions)
python train_cosmology.py --data-path ../datasets/cosmology --size smallest --batch-size 2 --use-wandb --experiment "scalar_mpnn_reldistrelposmv_pool" --num-samples 8192 --pooling-type RelDistRelPosMv --unpooling-type RelDistRelPosMv --mpnn-type scalar_only --mp-steps 3