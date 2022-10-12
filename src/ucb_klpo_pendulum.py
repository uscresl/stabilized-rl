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
from gp_ucb_algo import GPUCBAlgo

@wrap_experiment
def ucb_klpo_pendulum(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv('InvertedDoublePendulum-v2')

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

    inner_algo = KLPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        discount=0.99,
        batch_size=10000)

    algo = GPUCBAlgo(inner_algo)

    trainer.setup(algo, env)
    trainer.train(n_epochs=10000)


if __name__ == '__main__':
    clize.run(ucb_klpo_pendulum)
