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

SAVED_PICK_PLACE_MODEL_PATH = "./data/experiments/tmp/MT_10_klpo_stbl/env=pick-place-v2_seed=5555_target-kl=0.0015_kl-loss-coeff-lr=3.0_kl-loss-coeff-momentum=0.99999_historic-buffer-size=32000_note=tuned_xppo/saved_model.zip"
SAVED_WINDOW_OPEN = "./data/experiments/tmp/MT_10_klpo_stbl/env=window-open-v2_seed=3333_target-kl=0.0015_kl-loss-coeff-lr=3.0_kl-loss-coeff-momentum=0.99999_historic-buffer-size=32000_note=tuned_xppo/saved_model.zip"


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
def xppo_tranfer_MT10(
    ctxt=None,
    env="HalfCheetah-v3",
    normalize_batch_advantage=True,
    n_steps=4096,
    gae_lambda=0.97,
    vf_arch=[64, 64],
    clip_grad_norm=False,
    total_steps=3_000_000,
    note="buffer_kl_loss",
    multi_step_trust_region=False,
    *,
    seed,
    target_kl: float,
    ent_coef: float,
    kl_loss_coeff_lr: float,
    kl_loss_coeff_momentum: float,
    kl_target_stat: str,
    max_path_length: int,
    optimize_log_loss_coeff: bool,
    historic_buffer_size: int,
    reset_optimizers: bool,
    second_penalty_loop: bool,
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
        optimize_log_loss_coeff=optimize_log_loss_coeff,
        historic_buffer_size=historic_buffer_size,
        reset_optimizers=reset_optimizers,
        second_penalty_loop=second_penalty_loop,
        multi_step_trust_region=multi_step_trust_region,
    )
    random_value_net_sd = model.policy.value_net.state_dict()
    model.set_parameters(SAVED_PICK_PLACE_MODEL_PATH)
    model.policy.value_net.load_state_dict(random_value_net_sd)
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
        target_kl: float,
        optimize_log_loss_coeff: bool,
        note: str,
        historic_buffer_size: int,
        ent_coef: float = 0.0,
        kl_loss_coeff_lr: float = 1e-3,
        kl_loss_coeff_momentum: float = 0.0,
        kl_target_stat: str = "mean",
        n_steps: int = 4096,
        total_steps: int = 3_000_000,
        multi_step_trust_region=False,
        reset_optimizers: bool,
        second_penalty_loop: bool,
    ):
        env, max_path_length = gen_env(env)
        xppo_tranfer_MT10(
            dict(log_dir=log_dir),
            seed=seed,
            env=env,
            target_kl=target_kl,
            note=note,
            ent_coef=ent_coef,
            kl_loss_coeff_lr=kl_loss_coeff_lr,
            kl_loss_coeff_momentum=kl_loss_coeff_momentum,
            optimize_log_loss_coeff=optimize_log_loss_coeff,
            kl_target_stat=kl_target_stat,
            n_steps=n_steps,
            max_path_length=max_path_length,
            total_steps=total_steps,
            historic_buffer_size=historic_buffer_size,
            reset_optimizers=reset_optimizers,
            second_penalty_loop=second_penalty_loop,
            multi_step_trust_region=multi_step_trust_region,
        )
