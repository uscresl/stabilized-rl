#!/usr/bin/env python3
import clize
import torch
import numpy as np
import random

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.torch.optimizers import MinibatchOptimizer
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from klpo import KLPO
from gp_ucb_algo import GPUCBAlgo
from ucb_backtrack_algo import UCBBacktrackAlgo


@wrap_experiment(name_parameters="all", prefix="experiment/ucb_backtracking_ppo_envs")
def ucb_backtracking_ppo_envs(
    ctxt,
    env_name,
    seed=1,
    pg_loss_type="kl_div",
    learning_rate=2.5e-4,
    note="increased_hparam_range",
):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv(env_name)

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(
        env.spec,
        hidden_sizes=[64, 64],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
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

    inner_algo = KLPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        discount=0.99,
        batch_size=10000,
        lr_loss_coeff=0.1,
        lr_sq_loss_coeff=0.1,
        pg_loss_type=pg_loss_type,
        learning_rate=learning_rate,
        center_adv=True,
    )

    algo = UCBBacktrackAlgo(inner_algo)

    trainer.setup(algo, env)
    trainer.train(n_epochs=1000)


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
            ucb_backtracking_ppo_envs(env_name=env_name, seed=seed)
