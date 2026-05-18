#!/usr/bin/env bash
# Train a (non-causal) DQN baseline on a single game.
#
# Variants:
#   - default                            : distillation from teacher under masked obs
#   - LSTM=1                             : LSTM-DQN variant
#   - NO_DISTILL=1                       : vanilla DQN, no teacher
#   - NO_MASK=1                          : full (unmasked) observations
#
# Usage:
#   scripts/train_dqn_baseline.sh <ENV> [SEED] [GPU_ID]
#
# Example:
#   LSTM=1 scripts/train_dqn_baseline.sh Pong 13579 0
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

EXTRA=()
[[ "${LSTM:-0}" == "1" ]]       && EXTRA+=(--use-lstm)
[[ "${NO_DISTILL:-0}" == "1" ]] && EXTRA+=(--no-distill)
[[ "${NO_MASK:-0}" == "1" ]]    && EXTRA+=(--no-mask)

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
    "${EXTRA[@]}"
