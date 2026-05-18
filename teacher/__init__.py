# The DIAMOND actor-critic teacher is taken from
# "Diffusion for World Modeling: Visual Details Matter in Atari"
# (https://github.com/eloialonso/diamond). The CleanRL Sebulba PPO IMPALA
# Atari teacher comes from https://github.com/vwxyzjn/cleanrl.

from .actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from .diamond import load_ckpt_cfg
from .utils import extract_state_dict
from .sebulba_ppo_envpool_impala_atari_wrapper import Actor, Critic, Network
from .sebulba import SebulbaTeacher
