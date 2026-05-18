import os
# See https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

from functools import partial

import jax
import flax
import numpy as np
import jax.numpy as jnp
from huggingface_hub import hf_hub_download

from .sebulba_ppo_envpool_impala_atari_wrapper import Actor, Critic, Network


class SebulbaTeacher:
    """Wrapper around the pretrained CleanRL Sebulba PPO Atari agents on HuggingFace."""

    def __init__(self, env_id, action_dim, num_envs, seed):
        # CleanRL/Envpool call the variant v5 but it is the NoFrameskip-v4 ALE rom.
        hf_repository = f"cleanrl/{env_id}-v5-sebulba_ppo_envpool_impala_atari_wrapper-seed1"
        model_path = hf_hub_download(
            repo_id=hf_repository,
            filename="sebulba_ppo_envpool_impala_atari_wrapper.cleanrl_model",
        )
        self.network = Network()
        self.actor = Actor(action_dim=action_dim)
        self.critic = Critic()
        key = jax.random.PRNGKey(seed)
        self.key, network_key, actor_key, critic_key = jax.random.split(key, 4)
        dummy_obs = np.zeros((num_envs, 4, 84, 84))
        self.network_params = self.network.init(network_key, dummy_obs)
        hidden = self.network.apply(self.network_params, dummy_obs)
        self.actor_params = self.actor.init(actor_key, hidden)
        self.critic_params = self.critic.init(critic_key, hidden)

        with open(model_path, "rb") as f:
            _, (self.network_params, self.actor_params, self.critic_params) = flax.serialization.from_bytes(
                (None, (self.network_params, self.actor_params, self.critic_params)),
                f.read(),
            )
        self._get_action_and_logits = _make_jit_get_action_and_logits(
            self.network, self.actor, self.network_params, self.actor_params,
        )

    def get_action_and_logits(self, next_obs: np.ndarray):
        action, soft_logits, self.key = self._get_action_and_logits(next_obs, self.key)
        return action, soft_logits


def _make_jit_get_action_and_logits(network, actor, network_params, actor_params):
    @partial(jax.jit)
    def _get_action_and_logits(next_obs: np.ndarray, key: jax.random.PRNGKey):
        hidden = network.apply(network_params, next_obs)
        logits = actor.apply(actor_params, hidden)
        # Gumbel-softmax categorical sample. See:
        # https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        new_key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        soft_logits = logits - jnp.log(-jnp.log(u))
        action = jnp.argmax(soft_logits, axis=1)
        return action, soft_logits, new_key

    return _get_action_and_logits
