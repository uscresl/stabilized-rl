#!/usr/bin/env python3
import clize
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, RaySampler
from garage.torch.algos import VPG
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.torch.optimizers import MinibatchOptimizer

from backtracking_algo import BacktrackingAlgo

@wrap_experiment
def backtracking_vpg_pendulum(ctxt=None, seed=1):
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

    policy_optimizer = MinibatchOptimizer(torch.optim.Adam,
                                          policy,
                                          max_optimization_epochs=1)

    inner_algo = VPG(
        env_spec=env.spec,
        policy=policy,
        policy_optimizer=policy_optimizer,
        value_function=value_function,
        sampler=sampler,
        discount=0.99,
        center_adv=False,
        batch_size=10000)

    algo = BacktrackingAlgo(inner_algo)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100)


if __name__ == '__main__':
    clize.run(backtracking_vpg_pendulum)
