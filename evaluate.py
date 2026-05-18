"""Aggregate evaluation over the five-seed runs produced by ``run.py``.

For each (game, algo) combination this script locates the latest log
directory per seed, walks the saved checkpoints between 0 and 1M env
steps in 50K increments, runs ``num_envs`` evaluation episodes per
checkpoint, and writes a CSV under ``~/explogs/distill_results/``.
"""

import os
import argparse

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from envs import make_atari_env
from teacher import load_ckpt_cfg
from student import DQN, LSTM_DQN, student_obs_mask


ALGO_SIG = {
    "ours": "dqn-causal-causalloss-sebulba_teacher-fullmask",
    "distill_no_causal": "dqn-baseline-sebulba_teacher-fullmask",
    "distill_no_causal_lstm": "lstm-dqn-baseline-sebulba_teacher-fullmask",
    "no_distill_no_causal": "dqn-baseline-nodistill-fullmask",
}

SEEDS = [13579, 37485, 50879, 87592, 48590]

GAMES = [
    "Amidar", "Asterix", "Boxing", "Breakout", "ChopperCommand",
    "Pong", "Qbert", "RoadRunner", "Gopher", "KungFuMaster",
    "Seaquest", "MsPacman",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained algorithm on Atari games.")
    parser.add_argument("--algo", type=str, required=True, choices=ALGO_SIG.keys())
    parser.add_argument("--games", nargs="+", default=GAMES, choices=GAMES,
                        help="games to evaluate (default: all 12)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--log-dir", type=str,
                        default=os.path.expanduser("~/explogs/distill/"),
                        help="directory containing training run subdirectories")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.expanduser("~/explogs/distill_results/"))
    parser.add_argument("--max-step", type=int, default=1_000_000)
    parser.add_argument("--step-interval", type=int, default=50_000)
    return parser.parse_args()


def find_latest_runs(base_log_dir: str, algo_sig: str, game: str):
    """Return {seed: latest_run_subpath} for runs matching algo_sig and game."""
    all_runs = []
    for path in os.listdir(base_log_dir):
        full = os.path.join(base_log_dir, path)
        if os.path.isdir(full) and "debug" not in path:
            all_runs.extend(os.path.join(path, sub) for sub in os.listdir(full))

    def match(run_path):
        if algo_sig not in run_path or game not in run_path:
            return False
        if "lstm" not in algo_sig and "lstm" in run_path:
            return False
        return True

    latest = {}
    dates = {}
    for p in filter(match, all_runs):
        date_parts = p.split("-")[0:3]
        seed = p.split("-")[-4]
        if seed in dates:
            if int("".join(date_parts)) > int("".join(dates[seed])):
                dates[seed] = date_parts
                latest[seed] = p
        else:
            dates[seed] = date_parts
            latest[seed] = p
    return latest


def evaluate_checkpoint(agent, ckpt_path, eval_env, obs_mask, seeds, use_lstm, device):
    agent.load_state_dict(torch.load(Path(ckpt_path), weights_only=True, map_location=device))
    agent.to(device)
    agent.eval()

    state, _ = eval_env.reset(seed=seeds)
    total_reward = np.zeros_like(seeds, dtype=np.float32)
    done_mask = np.ones_like(seeds, dtype=np.float32)
    hx_cx = None
    with torch.no_grad():
        while True:
            if use_lstm:
                out, hx_cx = agent(obs_mask(state[:, :4, :, :]), hx_cx)
            else:
                out = agent(obs_mask(state[:, :4, :, :]))
            action = out.max(1)[1].view(-1)
            state, reward, term, trunc, _ = eval_env.step(action)
            total_reward += (reward.cpu().numpy() * done_mask)
            done_mask = np.logical_and(done_mask, np.logical_not(term.cpu().numpy() | trunc.cpu().numpy()))
            if sum(done_mask) == 0:
                break
    return total_reward.mean()


def main():
    args = parse_args()
    algo = args.algo
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    for game in args.games:
        obs_mask = student_obs_mask(game)
        _, cfg = load_ckpt_cfg(game)
        runs_dict = find_latest_runs(args.log_dir, ALGO_SIG[algo], game)
        assert set(map(int, runs_dict.keys())) == set(SEEDS), \
            f"Missing seeds for {game}/{algo}: {runs_dict.keys()}"

        perf_dict = {}
        for seed, seed_path in runs_dict.items():
            perf_dict[seed] = {}
            seeds = [int(seed) + i for i in range(args.num_envs)]
            eval_env = make_atari_env(num_envs=args.num_envs, device=device, **cfg.env.test)
            obs_dim = eval_env.unwrapped.single_observation_space.shape
            act_dim = eval_env.unwrapped.single_action_space.n
            use_lstm = "lstm" in algo
            if use_lstm:
                agent = LSTM_DQN(obs_dim[0], obs_dim[1], act_dim, lstm_dim=512)
            else:
                agent = DQN(obs_dim[0], obs_dim[1], act_dim)

            ckpt_dir = os.path.join(args.log_dir, seed_path, "checkpoints")
            ckpt_list = sorted(
                os.listdir(ckpt_dir),
                key=lambda x: int(x.split(".")[0]) if "complete" not in x else 10_000_000,
            )

            best_score, best_step = 0.0, -1
            for ckpt in tqdm(ckpt_list, desc=f"{game}-{algo}-{seed}:"):
                if "complete" in ckpt or int(ckpt.split(".")[0]) > args.max_step:
                    break
                if int(ckpt.split(".")[0]) % args.step_interval != 0:
                    continue

                ckpt_path = os.path.join(ckpt_dir, ckpt)
                mean_reward = evaluate_checkpoint(
                    agent, ckpt_path, eval_env, obs_mask, seeds, use_lstm, device,
                )
                step = ckpt.split('.')[0]
                tqdm.write(f"Algo: {algo} | Game: {game} | Seed: {seed} | Step: {step} | Avg. Rewards: {mean_reward}")
                perf_dict[seed][step] = mean_reward
                if mean_reward > best_score:
                    best_score, best_step = mean_reward, step
            print(f"Algo: {algo} | Game: {game} | Seed: {seed} | Best step: {best_step} | Best score: {best_score}")

        num_values = {len(v) for v in perf_dict.values()}
        if len(num_values) != 1:
            raise ValueError(f"Inconsistent number of values across seeds in perf_dict: {num_values}")
        df = pd.DataFrame(perf_dict)
        df.to_csv(os.path.join(args.output_dir, f"{game}_{algo}.csv"), header=False)


if __name__ == "__main__":
    main()
