#!/usr/bin/env bash
# Run a training script sequentially over the five seeds reported in the paper.
#
# Usage:
#   scripts/sweep_seeds.sh <TRAIN_SCRIPT> <ENV> [GPU_ID] [extra args...]
#
# Example (Causal-DQN, all five seeds on GPU 0):
#   scripts/sweep_seeds.sh scripts/train_causal_dqn.sh Pong 0
set -euo pipefail

TRAIN_SCRIPT=${1:?path to training script required}
ENV=${2:?env required}
GPU_ID=${3:-0}
shift 3 || true

SEEDS=(13579 37485 50879 87592 48590)

for SEED in "${SEEDS[@]}"; do
    echo "===== Seed $SEED ====="
    "$TRAIN_SCRIPT" "$ENV" "$SEED" "$GPU_ID" "$@"
done
