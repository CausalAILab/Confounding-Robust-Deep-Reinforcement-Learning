#!/usr/bin/env bash
# Aggregate evaluation over the five-seed runs stored under ~/explogs/distill.
# See evaluate.py for the run-name patterns that identify each algorithm.
#
# Usage:
#   scripts/evaluate.sh <ALGO>
#
# ALGO is a key from ALGO_SIG in evaluate.py
# (e.g. ours, distill_no_causal, distill_no_causal_lstm, no_distill_no_causal).
set -euo pipefail

ALGO=${1:?algo required}
python evaluate.py --algo "$ALGO"
