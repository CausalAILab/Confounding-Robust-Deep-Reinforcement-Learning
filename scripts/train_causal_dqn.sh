#!/usr/bin/env bash
# Train Causal-DQN (the method proposed in the paper) on a single game.
#
# Usage:
#   scripts/train_causal_dqn.sh <ENV> [SEED] [GPU_ID]
#
# Example:
#   scripts/train_causal_dqn.sh Pong 13579 0
set -euo pipefail

ENV=${1:-Pong}
SEED=${2:-13579}
GPU_ID=${3:-0}

LOG_DIR=${LOG_DIR:-$HOME/explogs/distill}
TEACHER=${TEACHER:-sebulba}
NUM_STEPS=${NUM_STEPS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-0.0005}
TAU=${TAU:-0.0001}
EPS_DECAY=${EPS_DECAY:-200000}
BOUND_MODE=${BOUND_MODE:-4}

python -u run.py \
    --env "$ENV" \
    --seed "$SEED" \
    --device-id "$GPU_ID" \
    --base-log-dir "$LOG_DIR" \
    --total-timesteps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LR" \
    --tau "$TAU" \
    --eps_decay "$EPS_DECAY" \
    --teacher "$TEACHER" \
    --causal \
    --bound-mode "$BOUND_MODE" \
    --causal-loss
