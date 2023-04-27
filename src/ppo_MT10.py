#!/usr/bin/env python3
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, VecWorker, DefaultWorker
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from metaworld.envs.mujoco.env_dict import MT10_V2

import clize


def gen_env(env: str):
    env_cls = MT10_V2[env]
    expert_env = env_cls()
    expert_env._partially_observable = False
    expert_env._set_task_called = True
    expert_env._freeze_rand_vec = False
    expert_env.reset()
    max_path_length = expert_env.max_path_length
    expert_env = GymEnv(expert_env, max_episode_length=max_path_length)
    assert max_path_length is not None
    return expert_env, max_path_length


@wrap_experiment(use_existing_dir=True)
def ppo_MT10(
    ctxt=None,
    *,
    seed,
    env,
    center_adv,
    total_steps,
    batch_size,
    gae_lambda=0.95,
    entropy_method="max",
    policy_ent_coeff=0.01,
    normalize_env=True,
    use_vec_worker=False,
    stop_entropy_gradient=False,
    note=None,
):
    set_seed(seed)
    env, max_path_length = gen_env(env)
    if normalize_env:
        env = normalize(env, normalize_obs=True, normalize_reward=True)

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(
        env.spec,
        hidden_sizes=[128, 128],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    sampler = LocalSampler(
        agents=policy,
        worker_class=VecWorker if use_vec_worker else DefaultWorker,
        envs=env,
        max_episode_length=max_path_length,
    )
    assert env.spec.max_episode_length is not None
    algo = PPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        discount=0.99,
        center_adv=center_adv,
        batch_size=batch_size,
        entropy_method=entropy_method,
        policy_ent_coeff=policy_ent_coeff,
        gae_lambda=gae_lambda,
        stop_entropy_gradient=stop_entropy_gradient,
    )
    n_epochs = total_steps // batch_size
    trainer.setup(algo, env)
    trainer.train(n_epochs=n_epochs)


if __name__ == "__main__":

    @clize.run
    def main(
        *,
        seed: int,
        env: str,
        batch_size: int,
        total_steps: int,
        log_dir: str,
        note: str,
        center_adv: bool,
        normalize_env: bool,
        entropy_method: str,
        policy_ent_coeff: float,
        gae_lambda: float,
        stop_entropy_gradient: bool,
        use_vec_worker: bool = False,
    ):
        ppo_MT10(
            dict(log_dir=log_dir),
            seed=seed,
            env=env,
            total_steps=total_steps,
            batch_size=batch_size,
            center_adv=center_adv,
            normalize_env=normalize_env,
            use_vec_worker=use_vec_worker,
            stop_entropy_gradient=stop_entropy_gradient,
            note=note,
            entropy_method=entropy_method,
            policy_ent_coeff=policy_ent_coeff,
            gae_lambda=gae_lambda,
        )
