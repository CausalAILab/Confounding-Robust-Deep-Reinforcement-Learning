import torch
from abc import ABC, abstractmethod
from typing import Callable

from constants import ATARI_100K_GAMES


class AgentInterface(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def estimate_value(self, inputs):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def train(self, optimizer, log_dir):
        pass

    @abstractmethod
    def evaluate(self, ep, visualize):
        pass


def student_obs_mask(env_name: str) -> Callable:
    """Return the per-game observation mask used to construct confounded inputs."""
    assert env_name in ATARI_100K_GAMES, f"'{env_name}' is not in the supported game list!"
    if env_name == "Pong":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, :, :15] = 0
            obs[:, :, :8, :] = 0
            return obs
    elif env_name == "Amidar":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, 53:, :] = 0
            return obs
    elif env_name == "Asterix":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, 50:, :] = 0
            return obs
    elif env_name == "Boxing":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, 56:, :] = 0
            obs[:, :, :8, :] = 0
            obs[:, :, :, 30:] = 0
            obs[:, :, :, :10] = 0
            return obs
    elif env_name == "Breakout":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, :28, :] = 0
            return obs
    elif env_name == "ChopperCommand":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, 52:, :] = 0
            obs[:, :, :15, :] = 0
            return obs
    elif env_name == "Gopher":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, :10, :] = 0
            obs[:, :, 55:, :] = 0
            return obs
    elif env_name == "KungFuMaster":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, :8, :] = 0
            return obs
    elif env_name == "MsPacman":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, 53:, :] = 0
            return obs
    elif env_name == "Qbert":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            return obs
    elif env_name == "RoadRunner":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, 58:, :] = 0
            obs[:, :, :20, :] = 0
            return obs
    elif env_name == "Seaquest":
        def helper(obs: torch.Tensor) -> torch.Tensor:
            obs[:, :, 58:, :] = 0
            obs[:, :, :6, :] = 0
            return obs
    else:
        raise NotImplementedError(f"Obs mask for '{env_name}' is not implemented yet.")
    return helper
