#!/usr/bin/env python3
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler, LocalSampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import clize


@wrap_experiment(use_existing_dir=True)
def ppo_mujoco(ctxt=None, *, seed, env, center_adv):
    set_seed(seed)
    env = GymEnv(env)

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               center_adv=center_adv,
               batch_size=10000)

    trainer.setup(algo, env)
    trainer.train(n_epochs=1000)


@clize.run
def main(*, seed: int, env: str, log_dir: str, center_adv: bool=True):
    ppo_mujoco(dict(log_dir=log_dir), seed=seed, env=env, center_adv=center_adv)
