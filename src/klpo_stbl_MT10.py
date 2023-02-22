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
def klpo_stbl_MT10(
    ctxt=None,
    env="HalfCheetah-v3",
    normalize_batch_advantage=True,
    n_steps=4096,
    gae_lambda=0.97,
    vf_arch=[64, 64],
    clip_grad_norm=False,
    total_steps=3_000_000,
    note="buffer_kl_loss",
    *,
    seed,
    target_kl,
    ent_coef,
    kl_loss_coeff_lr,
    kl_loss_coeff_momentum: float,
    kl_target_stat,
    max_path_length,
):
    model = KLPOStbl(
        "MlpPolicy",
        env,
        seed=seed,
        normalize_batch_advantage=normalize_batch_advantage,
        n_steps=n_steps,
        policy_kwargs={"net_arch": [{"vf": vf_arch, "pi": [64, 64]}]},
        gae_lambda=gae_lambda,
        clip_grad_norm=clip_grad_norm,
        target_kl=target_kl,
        ent_coef=ent_coef,
        kl_loss_coeff_lr=kl_loss_coeff_lr,
        kl_target_stat=kl_target_stat,
        kl_loss_coeff_momentum=kl_loss_coeff_momentum,
        max_path_length=max_path_length,
    )

    new_logger = configure(ctxt.snapshot_dir, ["stdout", "log", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_steps)


if __name__ == "__main__":

    @clize.run
    def main(
        *,
        seed: int,
        env: str,
        log_dir: str,
        target_kl: float,
        note: str,
        ent_coef: float = 0.0,
        kl_loss_coeff_lr: float = 1e-3,
        kl_loss_coeff_momentum: float = 0.0,
        kl_target_stat: str = "mean",
        n_steps: int = 4096,
    ):
        env, max_path_length = gen_env(env)
        n_steps = max_path_length
        klpo_stbl_MT10(
            dict(log_dir=log_dir),
            seed=seed,
            env=env,
            target_kl=target_kl,
            note=note,
            ent_coef=ent_coef,
            kl_loss_coeff_lr=kl_loss_coeff_lr,
            kl_loss_coeff_momentum=kl_loss_coeff_momentum,
            kl_target_stat=kl_target_stat,
            n_steps=n_steps,
            max_path_length=max_path_length,
        )
