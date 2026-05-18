import os
import copy
import random
import numpy as np
import gymnasium as gym
from collections import namedtuple, deque
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from constants import GAMES_REWARD_LB
from .utils import AgentInterface, student_obs_mask
from teacher import ActorCritic, SebulbaTeacher
from teacher.utils import init_lstm


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'action_prob', 'hx', 'cx'))


REWARD_CLIP_GAMES = {
    'Breakout', 'MsPacman', 'ChopperCommand', 'Gopher', 'RoadRunner',
    'Asterix', 'Amidar', 'KungFuMaster', 'Seaquest',
}

NEG_FLOOR_GAMES = {
    'Breakout', 'KungFuMaster', 'MsPacman', 'Seaquest', 'ChopperCommand',
    'Qbert', 'Gopher', 'RoadRunner', 'Asterix', 'Amidar',
}


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done, action_prob, hx=None, cx=None):
        if hx is None:
            if action is None:
                for i in range(state.shape[0]):
                    self.memory.append(Transition(state[i], None, None, None, None, action_prob[i], None, None))
            else:
                for i in range(state.shape[0]):
                    self.memory.append(Transition(state[i], action[i], reward[i], next_state[i], done[i], action_prob[i], None, None))
        else:
            if action is None:
                for i in range(state.shape[0]):
                    self.memory.append(Transition(state[i], None, None, None, None, action_prob[i], None, None))
            else:
                for i in range(state.shape[0]):
                    self.memory.append(Transition(state[i], action[i], reward[i], next_state[i], done[i], action_prob[i], hx[i], cx[i]))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.action_dim = action_dim

        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - kernel_size) // stride + 1
        convw, convh = w, h
        for k, s in [(8, 4), (4, 2), (3, 1)]:
            convw = conv2d_size_out(convw, kernel_size=k, stride=s)
            convh = conv2d_size_out(convh, kernel_size=k, stride=s)
        self.final_size = convw * convh * 128
        self.linear1 = nn.Linear(self.final_size, 512)
        self.linear2 = nn.Linear(512, self.action_dim)

    def forward(self, obs: Tensor) -> Tensor:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        action_feature = F.relu(self.linear1(x.flatten(start_dim=1)))
        return self.linear2(action_feature)


class LSTM_DQN(nn.Module):
    def __init__(self, h, w, action_dim, lstm_dim):
        super(LSTM_DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.action_dim = action_dim

        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - kernel_size) // stride + 1
        convw, convh = w, h
        for k, s in [(8, 4), (4, 2), (3, 1)]:
            convw = conv2d_size_out(convw, kernel_size=k, stride=s)
            convh = conv2d_size_out(convh, kernel_size=k, stride=s)
        self.lstm_input_size = convw * convh * 128
        self.lstm_hidden_size = lstm_dim
        self.lstm = nn.LSTMCell(self.lstm_input_size, self.lstm_hidden_size)
        self.head = nn.Linear(self.lstm_hidden_size, self.action_dim)
        init_lstm(self.lstm)

    def forward(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x = F.relu(self.bn1(self.conv1(obs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        hx, cx = self.lstm(x.flatten(start_dim=1), hx_cx)
        return self.head(hx), (hx, cx)


def _init_hx_cx(state, hidden_size):
    zero = torch.zeros([state.shape[0], hidden_size], device=state.device)
    return zero, zero.clone()


def _soft_update(target_net, source_net, tau):
    target_sd = target_net.state_dict()
    source_sd = source_net.state_dict()
    for key in source_sd:
        target_sd[key] = source_sd[key] * tau + target_sd[key] * (1 - tau)
    target_net.load_state_dict(target_sd)


class DQNInterface(AgentInterface):
    """Causal-DQN / DQN learner (with optional teacher distillation and LSTM backbone)."""

    def __init__(self, teacher: Union[ActorCritic, DQN], env: gym.vector.VectorEnv, eval_env: gym.vector.VectorEnv, args):
        self.env = env
        self.eval_env = eval_env
        self.args = args
        self.obs_dim = self.eval_env.unwrapped.single_observation_space.shape
        self.act_dim = self.eval_env.unwrapped.single_action_space.n
        self.device = torch.device(args.device)
        self.agent_obs_mask = (lambda x: x) if args.no_mask else student_obs_mask(args.env)
        if args.use_lstm:
            self.agent = LSTM_DQN(self.obs_dim[0], self.obs_dim[1], self.act_dim, args.lstm_dim).to(self.device)
            self.target_net = LSTM_DQN(self.obs_dim[0], self.obs_dim[1], self.act_dim, args.lstm_dim).to(self.device)
        else:
            self.agent = DQN(self.obs_dim[0], self.obs_dim[1], self.act_dim).to(self.device)
            self.target_net = DQN(self.obs_dim[0], self.obs_dim[1], self.act_dim).to(self.device)
        self.target_net.load_state_dict(self.agent.state_dict())
        if self.args.no_distill:
            self.behavioral = None
        elif args.teacher != 'sebulba':
            self.behavioral = teacher.to(self.device)
        else:
            self.behavioral = teacher

    def parameters(self):
        return self.agent.parameters()

    def estimate_value(self, obs_array, prev_act=None):
        return np.squeeze(self.agent.get_value(torch.from_numpy(obs_array).to(self.device)).detach().cpu().numpy())

    def load(self, agent_path):
        self.agent.load_state_dict(torch.load(agent_path, map_location=torch.device(self.device), weights_only=True))
        self.agent.eval()

    def evaluate(self, ep=1, visualize=False):
        env = self.eval_env
        hx_cx = None
        with torch.no_grad():
            total_reward = 0
            obs_seq = []
            for _ in range(ep):
                state, _ = env.reset()
                if visualize and ep == 1:
                    obs_seq.append(state[:, 4:, :, :].add(1).div(2).mul(255).type(torch.uint8).contiguous())
                while True:
                    if self.args.use_lstm:
                        out, hx_cx = self.agent(self.agent_obs_mask(state[:, 4:, :, :]), hx_cx)
                    else:
                        out = self.agent(self.agent_obs_mask(state[:, :4, :, :]))
                    action = out.max(1)[1].view(-1)
                    state, reward, term, trunc, _ = env.step(action)
                    if visualize and ep == 1:
                        obs_seq.append(state[:, 4:, :, :].add(1).div(2).mul(255).type(torch.uint8).contiguous())
                    total_reward += reward
                    if term or trunc:
                        break
        return (total_reward * 1.0 / ep).item(), obs_seq

    def train(self, optimizer: torch.optim, base_log_dir: str, scheduler: torch.optim.lr_scheduler.LRScheduler = None, eps_end: float = .1, eps_start: float = 1.0, eps_decay: int = 20000):
        steps_done = 0
        memory = ReplayMemory(self.args.buffer_size)
        writer = SummaryWriter(base_log_dir)
        check_point_path = os.path.join(base_log_dir, 'checkpoints/')
        writer.add_text("train/eps-decay", f"{eps_decay}", 0)
        os.makedirs(check_point_path, exist_ok=True)
        state, info = self.env.reset()
        cur_epi_rewards = torch.zeros(state.shape[0]).to(state.device)
        hx_cx = _init_hx_cx(state, self.agent.lstm_hidden_size) if self.args.use_lstm else None

        while steps_done < self.args.total_timesteps:
            self.agent.train()
            prev_hx_cx = copy.copy(hx_cx)
            sample = random.random()
            eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
            if sample > eps_threshold:
                with torch.no_grad():
                    if self.args.no_distill:
                        if self.args.use_lstm:
                            out, hx_cx = self.agent(self.agent_obs_mask(state[:, 4:, :, :]), hx_cx)
                        else:
                            out = self.agent(self.agent_obs_mask(state[:, :4, :, :]))
                        action = torch.max(out, dim=1)[1].view(-1)
                        action_prob = torch.ones_like(action, device=self.device)
                    elif self.args.teacher == "diamond":
                        logits_act, _, hx_cx = self.behavioral.predict_act_value(state[:, 4:, :, :], hx_cx)
                        dst = torch.distributions.categorical.Categorical(logits=logits_act)
                        action = dst.sample()
                        action_prob = torch.exp(dst.log_prob(action))
                    elif self.args.teacher == "sebulba":
                        self.behavioral: SebulbaTeacher
                        action, soft_logits = self.behavioral.get_action_and_logits(info["sebulba_obs"])
                        action = torch.tensor(np.array(action), device=self.device, dtype=torch.int64)
                        soft_logits = torch.tensor(np.array(soft_logits), device=self.device)
                        dst = torch.distributions.categorical.Categorical(logits=soft_logits)
                        action_prob = torch.exp(dst.log_prob(action)).to(self.device)
                    else:
                        out = self.behavioral(state[:, :4, :, :])
                        action = torch.max(out, dim=1)[1].view(-1)
                        action_prob = torch.ones_like(action, device=self.device)
            else:
                action = torch.randint(0, self.agent.action_dim, size=(state.shape[0],), dtype=torch.int64, device=self.device)
                action_prob = torch.ones(size=(state.shape[0],), device=self.device) / self.agent.action_dim
                hx_cx = _init_hx_cx(state, self.agent.lstm_hidden_size) if self.args.use_lstm else None

            steps_done += self.args.num_envs
            next_state, reward, term, trunc, info = self.env.step(action)
            cur_epi_rewards += reward

            done = torch.logical_or(term, trunc)
            if torch.any(done):
                avg_epi_reward = cur_epi_rewards[done].mean().item()
                writer.add_scalar('Train/Current Epi Reward', avg_epi_reward, steps_done)
                writer.add_scalar('Train/Progress', steps_done / self.args.total_timesteps, steps_done)
                writer.flush()
                cur_epi_rewards[done] = 0

            if self.args.env in REWARD_CLIP_GAMES:
                reward = torch.clamp(reward, -1.0, 1.0)
            if self.args.use_lstm:
                memory.push(
                    state[:, 4:, :, :].to('cpu'), action.to('cpu'), reward.to('cpu'),
                    next_state[:, 4:, :, :].to('cpu'), done.to('cpu'), action_prob.to('cpu'),
                    prev_hx_cx[0].to('cpu'), prev_hx_cx[1].to('cpu'),
                )
            else:
                memory.push(
                    state[:, :4, :, :].to('cpu'), action.to('cpu'), reward.to('cpu'),
                    next_state[:, :4, :, :].to('cpu'), done.to('cpu'), action_prob.to('cpu'),
                )
            writer.add_scalar('Train/Replay Buffer Size', len(memory), steps_done)
            writer.flush()

            state = next_state

            if len(memory) < self.args.batch_size or len(memory) < self.args.start_training_size:
                continue
            for _ in range(self.args.num_updates):
                transitions = memory.sample(self.args.batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.logical_not(torch.tensor(batch.done)).to(self.device)
                non_final_next_states = torch.stack(batch.next_state).to(self.device)[non_final_mask]
                state_batch = torch.stack(batch.state).to(self.device)
                action_batch = torch.tensor(batch.action, device=self.device)
                reward_batch = torch.tensor(batch.reward, device=self.device)
                action_prob_batch = torch.tensor(batch.action_prob, device=self.device)
                if self.args.use_lstm:
                    hx_batch = torch.stack(batch.hx).to(self.device)
                    cx_batch = torch.stack(batch.cx).to(self.device)
                neg_action_mask = torch.ones([self.args.batch_size, self.agent.action_dim], dtype=torch.bool).to(self.device)
                neg_action_mask[torch.arange(self.args.batch_size), action_batch] = 0
                neg_action_batch = torch.tensor(
                    [i for i in range(self.agent.action_dim)] * self.args.batch_size,
                    device=self.device,
                ).view([self.args.batch_size, -1])[neg_action_mask].flatten()

                if self.args.use_lstm:
                    full_state_action_values, hx_cx_pred = self.agent(self.agent_obs_mask(state_batch), (hx_batch, cx_batch))
                else:
                    full_state_action_values = self.agent(self.agent_obs_mask(state_batch))
                state_action_values = full_state_action_values.gather(dim=1, index=action_batch.unsqueeze(1))
                assert neg_action_batch.shape[0] == self.args.batch_size * (self.agent.action_dim - 1)
                neg_state_action_values = full_state_action_values.repeat_interleave(
                    self.agent.action_dim - 1, dim=0
                ).gather(dim=1, index=neg_action_batch.unsqueeze(1))

                next_state_values = torch.zeros(self.args.batch_size, device=self.device)
                with torch.no_grad():
                    if self.args.use_lstm:
                        non_final_hx_cx_pred = (hx_cx_pred[0][non_final_mask], hx_cx_pred[1][non_final_mask])
                        next_state_actions = self.agent(self.agent_obs_mask(non_final_next_states), non_final_hx_cx_pred)[0].max(dim=1, keepdim=True)[1].detach()
                        next_state_values[non_final_mask] = self.target_net(self.agent_obs_mask(non_final_next_states), non_final_hx_cx_pred)[0].gather(dim=1, index=next_state_actions).detach().squeeze()
                    else:
                        next_state_actions = self.agent(self.agent_obs_mask(non_final_next_states)).max(dim=1, keepdim=True)[1]
                        next_state_values[non_final_mask] = self.target_net(self.agent_obs_mask(non_final_next_states)).gather(dim=1, index=next_state_actions).squeeze()

                expected_state_action_values = (next_state_values * self.args.gamma) + reward_batch
                if self.args.causal:
                    neg_expected_state_action_values = _causal_lower_bound(
                        self.args, reward_batch, next_state_values
                    )
                    if self.args.bound_mode == 5:
                        expected_state_action_values = expected_state_action_values + neg_expected_state_action_values
                    else:
                        expected_state_action_values = (
                            action_prob_batch * expected_state_action_values
                            + (1 - action_prob_batch) * neg_expected_state_action_values
                        )

                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                if self.args.causal_loss:
                    loss += criterion(
                        neg_state_action_values,
                        torch.ones_like(neg_state_action_values, device=self.device) * neg_expected_state_action_values,
                    )

                optimizer.zero_grad()
                loss.backward()
                if self.args.use_lstm:
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_value_(self.agent.parameters(), 100)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Train/LearningRate", current_lr, steps_done)

                _soft_update(self.target_net, self.agent, self.args.tau)

            writer.add_scalar("Train/Loss", loss, steps_done)

            if steps_done % self.args.log_interval == 0 or self.args.debug:
                self.agent.eval()
                eva_reward, imgs = self.evaluate(1, visualize=True)
                torch.save(self.agent.state_dict(), os.path.join(check_point_path, f'{steps_done}.pt'))
                print('Checkpoint for step {} saved.\tEva Rewards: {:.2f}'.format(steps_done, eva_reward))
                writer.add_scalar('Evaluate/Avg Eva Reward', eva_reward, steps_done)
                writer.add_video('Evaluate/Visualization', torch.stack(imgs).permute((1, 0, 2, 3, 4)).cpu(), steps_done, fps=15)
                writer.flush()

            if self.args.debug:
                break

        torch.save(self.agent.state_dict(), os.path.join(check_point_path, 'complete.pt'))
        print('Complete')
        writer.close()


def _causal_lower_bound(args, reward_batch, next_state_values):
    """Worst-case Q-bound for the off-action update; see Algorithm 1 of the paper."""
    mode = args.bound_mode
    if mode == 0:
        return torch.max(reward_batch).item() + args.gamma * torch.max(next_state_values).item()
    if mode == 1:
        return torch.max(reward_batch).item() + args.gamma * torch.mean(next_state_values).item()
    if mode == 2:
        return torch.mean(reward_batch).item() + args.gamma * torch.mean(next_state_values).item()
    if mode == 3:
        return torch.mean(reward_batch).item() + args.gamma * torch.max(next_state_values).item()
    if mode == 4:
        bound = torch.min(reward_batch).item() + args.gamma * torch.min(next_state_values).item()
        if args.env in NEG_FLOOR_GAMES:
            bound = min(bound, -1.0)
        return bound
    return torch.ones_like(reward_batch) * GAMES_REWARD_LB[args.env]
