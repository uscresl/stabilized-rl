import clize

from garage import wrap_experiment
from torch.distributions.kl import (
    _kl_lowrankmultivariatenormal_lowrankmultivariatenormal,
)
from garage.envs import GymEnv, normalize
from klpo_stable_baselines_algo import KLPOStbl
from stable_baselines3.common.logger import configure
import random
from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerReachEnvV2
from metaworld.envs.mujoco.env_dict import MT10_V2
import os
from ppo_stable_baselines_algo import PPO


def gen_env(env: str):
    env_cls = MT10_V2[env]
    expert_env = env_cls()
    expert_env._partially_observable = False
    expert_env._set_task_called = True
    expert_env._freeze_rand_vec = False
    expert_env.reset()
    max_path_length = expert_env.max_path_length

    return expert_env, max_path_length


@wrap_experiment(name_parameters="all", use_existing_dir=True)
def ppo_stbl_MT10(
    ctxt=None,
    total_steps: int = 20_000_000,
    n_steps: int = 10_000,
    batch_size=32,
    gae_lambda: float = 1,
    note: str = "buffer_kl_loss",
    *,
    env: str,
    max_path_length: int,
    seed: int,
    gamma: float,
    n_epochs,
    learning_rate,
):
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        gae_lambda=gae_lambda,
        gamma=gamma,
        max_path_length=max_path_length,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
    )

    new_logger = configure(ctxt.snapshot_dir, ["stdout", "log", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_steps)
    model.save(
        os.path.join(ctxt.snapshot_dir, "saved_model.zip"), exclude=["historic_buffer"]
    )


if __name__ == "__main__":

    @clize.run
    def main(
        *,
        seed: int,
        env: str,
        log_dir: str,
        n_steps: int,
        gamma: float,
        note: str,
        learning_rate: float,
        n_epochs: int,
        batch_size=32,
        gae_lambda: float = 1,
        total_steps: int = 20_000_000,
    ):
        env, max_path_length = gen_env(env)
        ppo_stbl_MT10(
            dict(log_dir=log_dir),
            seed=seed,
            env=env,
            gamma=gamma,
            n_steps=n_steps,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
            note=note,
            max_path_length=max_path_length,
            total_steps=total_steps,
        )
