#!/usr/bin/env python3
import random

import clize
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.algos import PPO
from garage.torch.optimizers import MinibatchOptimizer
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.envs.normalized_env import NormalizedEnv
from klpo import KLPO


@wrap_experiment(name_parameters="all", prefix="experiment/ppo_baselines")
def ppo_multi_envs(
    ctxt,
    env_name,
    pg_loss_type: str = "kl_div",
    seed: int = 1,
    normalize_pg_loss: bool = False,
    target_lr=1.0,
    learning_rate=2.5e-4,
    note="pg_loss=ratio*adv",
):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = NormalizedEnv(GymEnv(env_name), normalize_obs=True, normalize_reward=True)

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(
        env.spec,
        hidden_sizes=[64, 64],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
        min_std=None,
    )

    value_function = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    sampler = LocalSampler(
        agents=policy, envs=env, max_episode_length=env.spec.max_episode_length
    )

    algo = PPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        center_adv=True,
        discount=0.99,
        batch_size=10000,
    )
    trainer.setup(algo, env)
    trainer.train(n_epochs=150)


if __name__ == "__main__":
    ppo_env_names = [
        "HalfCheetah-v3",
        "Hopper-v3",
        "Walker2d-v3",
        "Swimmer-v3",
        "InvertedPendulum-v2",
        "Reacher-v2",
    ]
    for env_name in ppo_env_names:
        for _ in range(5):
            seed = random.randrange(1000)
            ppo_multi_envs(env_name=env_name, seed=seed)
