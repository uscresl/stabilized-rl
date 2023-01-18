import clize

from garage import wrap_experiment
from klpo_stable_baselines_algo import KPLOStbl
from stable_baselines3.common.logger import configure
import random


@wrap_experiment(name_parameters="all", prefix="experiment/stbl")
def kpo_stbl(
    ctxt=None,
    env="HalfCheetah-v3",
    normalize_batch_advantage=True,
    lr_loss_coeff=0.5,
    lr_sq_loss_coeff=0,
    seed=1,
):
    model = KPLOStbl(
        "MlpPolicy",
        env,
        lr_loss_coeff=lr_loss_coeff,
        lr_sq_loss_coeff=lr_sq_loss_coeff,
        seed=seed,
        normalize_batch_advantage=normalize_batch_advantage,
    )

    new_logger = configure(ctxt.snapshot_dir, ["stdout", "log", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(3_000_000)


if __name__ == "__main__":
    ppo_env_names = [
        "HalfCheetah-v3",
        "Walker2d-v3",
        "Hopper-v3",
        "Swimmer-v3",
        "InvertedPendulum-v2",
        "Reacher-v2",
    ]
    for _ in range(5):
        for env_name in ppo_env_names:
            for normalize_batch_advantage in [True, False]:
                seed = random.randrange(1000)
                kpo_stbl(
                    env=env_name,
                    seed=seed,
                    normalize_batch_advantage=normalize_batch_advantage,
                )
