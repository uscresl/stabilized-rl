#!/usr/bin/env python3
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, VecWorker, DefaultWorker
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import clize
from klpo import KLPO


@wrap_experiment(use_existing_dir=True)
def klpo_mujoco(ctxt=None, *, seed, env, lr_loss_coeff, lr_sq_loss_coeff,
                loss_type, normalize_env, use_vec_worker):
    set_seed(seed)
    env = GymEnv(env)
    if normalize_env:
        env = normalize(env, normalize_obs=True, normalize_reward=True)

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
                           worker_class=VecWorker if use_vec_worker else DefaultWorker,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length)

    algo = KLPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                sampler=sampler,
                lr_loss_coeff=lr_loss_coeff,
                lr_sq_loss_coeff=lr_sq_loss_coeff,
                pg_loss_type=loss_type,
                discount=0.99,
                center_adv=True,
                batch_size=10000)

    trainer.setup(algo, env)
    trainer.train(n_epochs=1000)


@clize.run
def main(*, seed: int, env: str, log_dir: str,
         lr_loss_coeff: float=0.1, lr_sq_loss_coeff: float=0.,
         loss_type: str="kl_div",
         normalize_env: bool=False, use_vec_worker: bool=False):
    klpo_mujoco(dict(log_dir=log_dir), seed=seed, env=env,
                lr_loss_coeff=lr_loss_coeff,
                lr_sq_loss_coeff=lr_sq_loss_coeff,
                loss_type=loss_type,
                normalize_env=normalize_env,
                use_vec_worker=use_vec_worker)
