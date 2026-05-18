import os
import torch
import json
import random
import argparse
import numpy as np

from datetime import datetime
from pathlib import Path
from hydra.utils import instantiate

from constants import ATARI_100K_GAMES
from envs import make_atari_env
from teacher import load_ckpt_cfg, extract_state_dict
from teacher import ActorCritic, SebulbaTeacher
from student import DQN, DQNInterface


def parse_args():
    parser = argparse.ArgumentParser("Causal Agent Distillation")
    parser.add_argument(
        "--base-log-dir",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "explogs/distill"),
        help="path to base log dir",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Pong",
        choices=ATARI_100K_GAMES,
        help="Atari game to run",
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="dqn",
        choices=["dqn"],
        help="student algorithm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        default=13579,
        const=13579,
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--device-id",
        type=str,
        default="0",
        help='GPU id(s), e.g., "0" or "0,1,3"',
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--torch-deterministic", action="store_true", default=False)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="if toggled, run a single training step and exit",
    )
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument(
        "--eva",
        action="store_true",
        default=False,
        help="evaluate checkpoints under EVA_PATH instead of training",
    )
    parser.add_argument("--eva-len", type=int, default=15)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="use the causal lower-bound Bellman update",
    )
    parser.add_argument(
        "--causal-loss",
        action="store_true",
        default=False,
        help="add the auxiliary causal loss for negative actions",
    )
    parser.add_argument(
        "--bound-mode",
        type=int,
        default=0,
        help="bound mode: 0-maxmax, 1-maxmean, 2-meanmean, 3-meanmax, 4-minmin, 5-game-specific lower bound",
    )
    parser.add_argument(
        "--teacher",
        choices=["diamond", "sebulba"],
        default="sebulba",
    )

    # DQN-specific arguments
    parser.add_argument("--tau", type=float, default=.002, help="soft target update rate")
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--start_training_size", type=int, default=50_000)
    parser.add_argument("--eps_decay", type=int, default=100_000)
    parser.add_argument("--no-distill", action="store_true", default=False, help="train a vanilla DQN with no teacher")
    parser.add_argument("--no-mask", action="store_true", default=False, help="use the original (unmasked) observation")
    parser.add_argument("--use-lstm", action="store_true", default=False, help="use the LSTM-DQN architecture")

    # Shared / PPO-like arguments (kept for forward compatibility)
    parser.add_argument("--anneal-lr", action="store_true", default=False)
    parser.add_argument("--gae", action="store_false", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--norm-adv", action="store_false", default=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_false", default=True)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--target-ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.01)

    return parser.parse_args()


def build_log_dir(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S")
    if args.debug:
        suffix = "debug-logs"
    elif args.eva:
        suffix = "eva-logs"
    else:
        suffix = "logs"
    method_name = (
        f"{args.env}-"
        f"{'lstm-' if args.use_lstm else ''}{args.learner}-"
        f"{'causal-' + str(args.bound_mode) + '-' if args.causal else 'baseline'}"
        f"{'-causalloss' if args.causal_loss else ''}-"
        f"{'nodistill' if args.no_distill else args.teacher + '_teacher'}"
        f"{'-nomask' if args.no_mask else '-fullmask'}-"
        f"{args.seed}-{args.batch_size}-{args.learning_rate}-{args.tau}"
    )
    return os.path.join(args.base_log_dir, f"{timestamp}-{suffix}", method_name)


def build_teacher(args, cfg, teacher_ckpt_path, obs_dim, act_dim):
    if args.no_distill:
        return None
    print(f"Loading teacher ({args.teacher}) for env {args.env}...")
    if args.teacher == "diamond":
        teacher_config = instantiate(cfg.agent.actor_critic, num_actions=act_dim)
        teacher = ActorCritic(teacher_config)
        args.lstm_dim = teacher_config.lstm_dim
        ckpt = torch.load(Path(teacher_ckpt_path), map_location=args.device, weights_only=True)
        ckpt_dict = {k: extract_state_dict(ckpt, k) for k in ("denoiser", "rew_end_model", "actor_critic")}
        teacher.load_state_dict(ckpt_dict["actor_critic"])
        return teacher
    if args.teacher == "sebulba":
        return SebulbaTeacher(args.env, act_dim, args.num_envs, args.seed)
    # Fallback: pretrained DQN teacher (Pong only).
    assert args.env == "Pong", "DQN teacher only supports Pong!"
    teacher = DQN(obs_dim[0], obs_dim[1], act_dim)
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"teacher/ckpt/{args.env}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint {ckpt_path} not found. Train a DQN teacher for {args.env} first."
        )
    teacher.load_state_dict(torch.load(Path(ckpt_path), weights_only=True))
    return teacher


def build_learner(args, teacher, env, eval_env):
    if args.learner != "dqn":
        raise NotImplementedError(f"Unsupported learner: {args.learner}")
    return DQNInterface(teacher, env, eval_env, args)


def main():
    args = parse_args()
    args.num_updates = 1
    args.lstm_dim = 512

    args.base_log_dir = build_log_dir(args)
    print(f"Log dir {args.base_log_dir}")
    os.makedirs(args.base_log_dir, exist_ok=True)
    with open(os.path.join(args.base_log_dir, "params.json"), "w") as f:
        json.dump(vars(args), f)

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.use_deterministic_algorithms(args.torch_deterministic)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    teacher_ckpt_path, cfg = load_ckpt_cfg(args.env)
    env = make_atari_env(num_envs=args.num_envs, device=args.device, **cfg.env.train)
    eval_env = make_atari_env(num_envs=1, device=args.device, **cfg.env.test)
    obs_dim = eval_env.unwrapped.single_observation_space.shape
    act_dim = eval_env.unwrapped.single_action_space.n

    teacher = build_teacher(args, cfg, teacher_ckpt_path, obs_dim, act_dim)
    learner = build_learner(args, teacher, env, eval_env)

    print(
        f"Env: {args.env} | Causal: {args.causal} | Student: {args.learner} | "
        f"Teacher: {args.teacher} | Total Env Steps: {args.total_timesteps} | "
        f"Num Updates / VecEnv Step: {args.num_updates}"
    )

    optimizer = torch.optim.AdamW(learner.agent.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_timesteps // args.num_envs, eta_min=1e-6
    )
    learner.train(optimizer, args.base_log_dir, scheduler, eps_decay=args.eps_decay)
    print("Training finished.")


if __name__ == "__main__":
    main()
