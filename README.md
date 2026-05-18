# Confounding Robust Deep Reinforcement Learning: A Causal Approach

Official implementation for the NeurIPS 2025 paper
[**"Confounding Robust Deep Reinforcement Learning: A Causal Approach"**](https://openreview.net/pdf?id=9fUr5iFU9j)
by Mingxuan Li, Junzhe Zhang, and Elias Bareinboim (Causal AI Lab, Columbia
University & Syracuse University).

## Overview

Standard off-policy deep RL algorithms such as DQN assume *no unmeasured
confounders* (NUC): every variable that influenced the data-collection
policy is recorded in the observation. In real demonstrations — videos,
human gameplay, log replays — that assumption is rarely true. Unobserved
confounders break the equivalence between the behavior and target
policies' Q-values and make the off-policy target value
*non-identifiable*.

This repository implements **Causal-DQN**, a robust off-policy deep RL
algorithm that handles confounded demonstrations by deriving a
worst-case (partially identified) lower bound on the next-state value and
plugging it into the Bellman update whenever the learner deviates from
the demonstrator's action. The result is a safe student policy that
remains competitive with — and on several games **surpasses** — its own
demonstrator, even when the demonstrator sees information the student
does not.

Empirically, we train students on twelve *confounded* Atari games in
which the agent's input is partially masked relative to the
demonstrator's. Causal-DQN dominates the standard DQN baselines and the
LSTM-DQN baseline across the suite.

## Repository layout

```
configs/        # Hydra / OmegaConf configs for the DIAMOND env wrapper
constants.py    # Per-game lower-reward bounds + Atari-100k game list
envs/           # Atari preprocessing (64x64 student / 84x84 teacher obs streams)
teacher/        # Two teacher backends:
                #   - DIAMOND  (Alonso et al., 2024 — actor-critic + ResNet)
                #   - Sebulba  (CleanRL PPO IMPALA — JAX/Flax)
student/        # DQN (Nature CNN + LSTM variants) and the Causal-DQN learner
run.py          # Single-run training entry point
evaluate.py     # Five-seed aggregation evaluator -> CSV
scripts/        # Convenience shell wrappers (parameterized)
```

## Installation

The code targets Python 3.10+ on Linux with CUDA. Install the system
ALE/SDL dependencies, then create a fresh environment:

```bash
# System (Ubuntu / Debian)
sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev \
                        libsdl2-ttf-dev libgl1 libglib2.0-0

# Python environment
conda create -n causal-dqn python=3.10 -y
conda activate causal-dqn

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install \
    "gymnasium[atari]==0.29.*" ale-py opencv-python \
    hydra-core omegaconf huggingface_hub \
    tensorboard tqdm pandas numpy scipy \
    "jax[cuda12]" flax
```

The DIAMOND and Sebulba teacher checkpoints are downloaded lazily from
the HuggingFace Hub the first time you call `run.py`; you do not need to
fetch them manually.

## Usage

All training is driven by `run.py`. The shell scripts under `scripts/`
expose the most useful presets via environment variables and positional
arguments.

### Causal-DQN (the proposed method)

```bash
# scripts/train_causal_dqn.sh <ENV> [SEED] [GPU_ID]
scripts/train_causal_dqn.sh Pong 13579 0
```

This trains Causal-DQN on `Pong` with the Sebulba PPO teacher, the
game-specific causal lower-bound reward floor (`--bound-mode 5`) and the
auxiliary causal loss (`--causal-loss`). Override defaults via env vars
(`NUM_STEPS`, `BATCH_SIZE`, `LR`, `TAU`, `EPS_DECAY`, `LOG_DIR`,
`TEACHER`).

### Non-causal DQN baselines

```bash
# Vanilla DQN distilled from the teacher (default CNN backbone)
scripts/train_dqn_baseline.sh Pong 13579 0

# LSTM-DQN variant
LSTM=1 scripts/train_dqn_baseline.sh Pong 13579 0

# Vanilla DQN with no teacher (online learning) and full (unmasked) obs
NO_DISTILL=1 NO_MASK=1 scripts/train_dqn_baseline.sh Pong 13579 0
```

### Five-seed sweep (paper protocol)

```bash
# Runs the 5 paper seeds (13579, 37485, 50879, 87592, 48590) sequentially.
scripts/sweep_seeds.sh scripts/train_causal_dqn.sh Pong 0
```

### Aggregating evaluation results

After the seed sweep finishes, `evaluate.py` walks each run's checkpoint
directory and produces one CSV per (game, algo) under
`~/explogs/distill_results/`:

```bash
scripts/evaluate.sh ours                     # Causal-DQN
scripts/evaluate.sh distill_no_causal        # vanilla DQN + distillation
scripts/evaluate.sh distill_no_causal_lstm   # LSTM-DQN + distillation
scripts/evaluate.sh no_distill_no_causal     # vanilla DQN, no teacher
```

See `ALGO_SIG` in `evaluate.py` for the full list of supported algorithm
keys.

### Hyperparameters

The defaults in `run.py` reproduce the protocol in Appendix D of the
paper:

| Parameter            | Value             |
|----------------------|-------------------|
| Total env steps      | 1,000,000         |
| Vectorized envs      | 20                |
| Batch size           | 512               |
| Replay buffer        | 100,000           |
| Learning rate        | 5e-4 (cosine→1e-6)|
| Optimizer            | AdamW             |
| Target soft-update τ | 1e-4              |
| ε-decay              | 200,000           |
| γ                    | 0.99              |
| Bound mode           | 5 (game floor)    |

Atari games covered (paper Table 1): `Amidar, Asterix, Boxing, Breakout,
ChopperCommand, Gopher, KungFuMaster, MsPacman, Pong, Qbert, RoadRunner,
Seaquest`. Per-game observation masks are defined in
[`student/utils.py`](student/utils.py).

## Acknowledgements

- DIAMOND actor-critic teacher and Atari preprocessing wrapper:
  Alonso et al., [*Diffusion for World Modeling: Visual Details Matter in Atari*](https://github.com/eloialonso/diamond).
- Sebulba PPO IMPALA Atari teacher: [CleanRL](https://github.com/vwxyzjn/cleanrl).

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{li2025confounding,
  title     = {Confounding Robust Deep Reinforcement Learning: A Causal Approach},
  author    = {Li, Mingxuan and Zhang, Junzhe and Bareinboim, Elias},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=9fUr5iFU9j}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
