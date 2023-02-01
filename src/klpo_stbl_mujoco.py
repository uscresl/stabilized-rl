import clize

from garage import wrap_experiment
from torch.distributions.kl import (
    _kl_lowrankmultivariatenormal_lowrankmultivariatenormal,
)
from klpo_stable_baselines_algo import KLPOStbl
from stable_baselines3.common.logger import configure
import random


@wrap_experiment(name_parameters="all", use_existing_dir=True)
def klpo_stbl(
    ctxt=None,
    env="HalfCheetah-v3",
    normalize_batch_advantage=True,
    lr_loss_coeff=0,
    lr_sq_loss_coeff=0,
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
):
    model = KLPOStbl(
        "MlpPolicy",
        env,
        lr_loss_coeff=lr_loss_coeff,
        lr_sq_loss_coeff=lr_sq_loss_coeff,
        seed=seed,
        normalize_batch_advantage=normalize_batch_advantage,
        n_steps=n_steps,
        policy_kwargs={"net_arch": [{"vf": vf_arch, "pi": [64, 64]}]},
        gae_lambda=gae_lambda,
        clip_grad_norm=clip_grad_norm,
        target_kl=target_kl,
        ent_coef=ent_coef,
        kl_loss_coeff_lr=kl_loss_coeff_lr,
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
    ):
        klpo_stbl(
            dict(log_dir=log_dir),
            seed=seed,
            env=env,
            target_kl=target_kl,
            note=note,
            ent_coef=ent_coef,
            kl_loss_coeff_lr=kl_loss_coeff_lr,
        )

    # ppo_env_names = [
    # "HalfCheetah-v3",
    # "Walker2d-v3",
    # "Hopper-v3",
    # "Swimmer-v3",
    # "InvertedPendulum-v2",
    # "Reacher-v2",
    # ]
    # # ppo_env_names = [
    # #     "HalfCheetah-v3",
    # # ]
    # lr_loss_coeffs = [1, 1.5, 2, 7]
    # for _ in range(5):
    # seed = random.randrange(1000)
    # for loss_coeff in lr_loss_coeffs:
    # for env_name in ppo_env_names:
    # klpo_stbl(
    # env=env_name,
    # lr_loss_coeff=loss_coeff,
    # seed=seed,
    # normalize_batch_advantage=True,
    # clip_grad_norm=True,
    # total_steps=2_000_000,
    # )
