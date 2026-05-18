import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf, DictConfig


def download(filename: str) -> Path:
    return Path(hf_hub_download(repo_id="eloialonso/diamond", filename=filename))


def load_ckpt_cfg(name: str) -> tuple:
    """Fetch the pretrained DIAMOND actor-critic checkpoint and the env config for ``name``."""
    path_ckpt = download(f"atari_100k/models/{name}.pt")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = DictConfig({})
    cfg.agent = OmegaConf.load(os.path.join(base_dir, "../configs/agent.yaml"))
    cfg.env = OmegaConf.load(os.path.join(base_dir, "../configs/atari.yaml"))
    cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
    return path_ckpt, cfg
