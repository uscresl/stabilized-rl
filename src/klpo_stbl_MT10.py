from typing import Optional
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
    *,
    env,
    normalize_batch_advantage,
    n_steps,
    total_steps,
    note,
    seed,
    batch_size,
    target_kl,
    ent_coef,
    kl_loss_coeff_lr,
    kl_loss_coeff_momentum,
    kl_target_stat,
    optimize_log_loss_coeff,
    reset_optimizers,
    minibatch_kl_penalty,
    use_beta_adam,
    sparse_second_loop,
    normalize_advantage,
    second_loop_batch_size,
    second_penalty_loop,
    historic_buffer_size,
    second_loop_vf,
    multi_step_trust_region,
    maximum_kl_loss_coeff,
    max_path_length,
    early_stop_epoch: Optional[bool] = False,
    early_stop_across_epochs: Optional[bool] = False,
    bang_bang_kl_loss_opt: Optional[bool] = False,
    bang_bang_reset_kl_loss_coeff: Optional[bool] = False,
    v_trace: bool = False,
    reset_beta: bool,
):
    model = KLPOStbl(
        "MlpPolicy",
        env,
        seed=seed,
        normalize_batch_advantage=normalize_batch_advantage,
        n_steps=n_steps,
        policy_kwargs={"net_arch": [{"vf": [128, 128], "pi": [128, 128]}]},
        max_path_length=max_path_length,
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=target_kl,
        ent_coef=ent_coef,
        batch_size=batch_size,
        kl_loss_coeff_lr=kl_loss_coeff_lr,
        kl_target_stat=kl_target_stat,
        kl_loss_coeff_momentum=kl_loss_coeff_momentum,
        optimize_log_loss_coeff=optimize_log_loss_coeff,
        reset_optimizers=reset_optimizers,
        minibatch_kl_penalty=minibatch_kl_penalty,
        use_beta_adam=use_beta_adam,
        sparse_second_loop=sparse_second_loop,
        normalize_advantage=normalize_advantage,
        second_loop_batch_size=second_loop_batch_size,
        second_penalty_loop=second_penalty_loop,
        historic_buffer_size=historic_buffer_size,
        second_loop_vf=second_loop_vf,
        multi_step_trust_region=multi_step_trust_region,
        maximum_kl_loss_coeff=maximum_kl_loss_coeff,
        eval_policy=False,
        early_stop_epoch=early_stop_epoch,
        early_stop_across_epochs=early_stop_across_epochs,
        bang_bang_kl_loss_opt=bang_bang_kl_loss_opt,
        bang_bang_reset_kl_loss_coeff=bang_bang_reset_kl_loss_coeff,
        v_trace=v_trace,
        reset_beta=reset_beta,
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
        target_kl: float,
        note: str,
        ent_coef: float = 0.0,
        kl_loss_coeff_lr: float = 3.0,
        kl_loss_coeff_momentum: float = 0.9999,  # Not used with adam
        kl_target_stat: str = "max",
        n_steps: int = 4096,
        batch_size: int = 512,
        optimize_log_loss_coeff: bool = False,
        reset_optimizers: bool = True,
        minibatch_kl_penalty: bool = True,
        use_beta_adam: bool = True,
        sparse_second_loop: bool = True,
        normalize_advantage: bool = False,
        second_penalty_loop: bool = True,
        total_steps: int = 3_000_000,
        second_loop_batch_size: int = 16000,
        historic_buffer_size: int = 32000,
        second_loop_vf: bool = False,
        multi_step_trust_region: bool = False,
        maximum_kl_loss_coeff: int = 2**20,
        early_stop_epoch: bool = False,
        early_stop_across_epochs: bool = False,
        bang_bang_kl_loss_opt: bool = False,
        bang_bang_reset_kl_loss_coeff: bool = False,
        v_trace: bool = False,
        reset_beta: bool = True,
    ):
        env, max_path_length = gen_env(env)
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
            batch_size=batch_size,
            optimize_log_loss_coeff=optimize_log_loss_coeff,
            reset_optimizers=reset_optimizers,
            minibatch_kl_penalty=minibatch_kl_penalty,
            use_beta_adam=use_beta_adam,
            sparse_second_loop=sparse_second_loop,
            normalize_advantage=normalize_advantage,
            normalize_batch_advantage=normalize_advantage,
            total_steps=total_steps,
            second_loop_batch_size=second_loop_batch_size,
            second_penalty_loop=second_penalty_loop,
            historic_buffer_size=historic_buffer_size,
            second_loop_vf=second_loop_vf,
            multi_step_trust_region=multi_step_trust_region,
            maximum_kl_loss_coeff=maximum_kl_loss_coeff,
            max_path_length=max_path_length,
            early_stop_epoch=early_stop_epoch,
            early_stop_across_epochs=early_stop_across_epochs,
            bang_bang_kl_loss_opt=bang_bang_kl_loss_opt,
            bang_bang_reset_kl_loss_coeff=bang_bang_reset_kl_loss_coeff,
            v_trace=v_trace,
            reset_beta=reset_beta,
        )
