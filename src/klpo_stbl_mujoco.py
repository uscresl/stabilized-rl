import clize

from garage import wrap_experiment
from klpo_stable_baselines_algo import KLPOStbl
from stable_baselines3.common.logger import configure
import random


@wrap_experiment(name_parameters="all", prefix="experiment/stbl")
def klpo_stbl(
    ctxt=None,
    env="HalfCheetah-v3",
    normalize_batch_advantage=True,
    lr_loss_coeff=5.0,
    lr_sq_loss_coeff=0,
    n_steps=4096,
    gae_lambda=0.97,
    vf_arch=[64, 64],
    clip_grad_norm=False,
    total_steps=3_000_000,
    seed=1,
    note="fixed_old_policy_update",
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
    )

    new_logger = configure(ctxt.snapshot_dir, ["stdout", "log", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_steps)


if __name__ == "__main__":
    ppo_env_names = [
        "HalfCheetah-v3",
        "Walker2d-v3",
        "Hopper-v3",
        "Swimmer-v3",
        "InvertedPendulum-v2",
        "Reacher-v2",
    ]
    # ppo_env_names = [
    #     "HalfCheetah-v3",
    # ]
    lr_loss_coeffs = [0.7, 1, 1.5, 2, 4, 5, 10]
    for _ in range(5):
        for env_name in ppo_env_names:
            for loss_coeff in lr_loss_coeffs:
                for clip_grad in [True, False]:
                    for normalize_batch_advantage in [True, False]:
                        seed = random.randrange(1000)
                        klpo_stbl(
                            env=env_name,
                            lr_loss_coeff=loss_coeff,
                            seed=seed,
                            normalize_batch_advantage=normalize_batch_advantage,
                            clip_grad_norm=clip_grad,
                            total_steps=2_000_000,
                        )
