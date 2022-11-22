#!/usr/bin/env python3
import clize
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.torch.optimizers import MinibatchOptimizer

from klpo import KLPO

@wrap_experiment(name_parameters="all")
def klpo_pendulum(ctxt=None, seed: int=1,
                  lr_clip_range: float=-1,
                  lr_loss_coeff: float=0.0,
                  lr_sq_loss_coeff: float=1e-3,
                  normalize_pg_loss: bool=False,
                  target_lr=1.,
                  learning_rate=2.5e-4,
                  ):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv('InvertedDoublePendulum-v2')
    if lr_clip_range <= 0:
        lr_clip_range = None
    if lr_loss_coeff <= 0:
        lr_loss_coeff = 0.

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None,
                               min_std=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length)

    algo = KLPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        lr_clip_range=lr_clip_range,
        lr_loss_coeff=lr_loss_coeff,
        lr_sq_loss_coeff=lr_sq_loss_coeff,
        normalize_pg_loss=normalize_pg_loss,
        target_lr=target_lr,
        learning_rate=learning_rate,
        center_adv=True,
        discount=0.99,
        batch_size=10000)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100)


if __name__ == '__main__':
    clize.run(klpo_pendulum)
