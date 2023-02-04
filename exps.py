from doexp import cmd, In, Out, GLOBAL_CONTEXT
from socket import gethostname
import random
import plot_all_csvs
import sys

# Uncomment to stop starting new runs
# sys.exit(1)

HOST = gethostname()

if HOST == "brain.usc.edu":
    GLOBAL_CONTEXT.max_concurrent_jobs = 8

mujoco_envs = [
    # "InvertedDoublePendulum-v2",
    "HalfCheetah-v2",
    # "Hopper-v2",
    "Walker2d-v2",
]
seeds = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]

if HOST == "brain.usc.edu":
    for seed in seeds:
        for env in mujoco_envs:
            cmd(
                "python",
                "src/ppo_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--center-adv=True",
                "--normalize-env",
                "--log-dir",
                Out(f"ppo/env={env}_seed={seed}_normalized/"),
                warmup_time=3,
                ram_gb=20,
                priority=-10,
            )

ppo_env_names_v3 = [
    "HalfCheetah-v3",
    "Walker2d-v3",
    "Hopper-v3",
    "Swimmer-v3",
    "InvertedPendulum-v2",
    "Reacher-v2",
]

if HOST == "resl34":
    GLOBAL_CONTEXT.max_concurrent_jobs = 12

for seed in seeds:
    target_kl = 0.2
    ram_gb = 4
    for env in mujoco_envs:
        cmd(
            "python",
            "src/klpo_stbl_mujoco.py",
            "--seed",
            seed,
            "--env",
            env,
            "--note",
            "learned-kl-loss",
            "--target-kl",
            target_kl,
            "--log-dir",
            Out(
                f"klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_note=learned-kl-loss/"
            ),
            warmup_time=3,
            ram_gb=ram_gb,
            priority=10,
        )
        for ent_coef in [0.01, 0.001]:
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--note",
                "learned-kl-loss",
                "--target-kl",
                target_kl,
                "--ent-coef",
                ent_coef,
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_ent-coef={ent_coef}_note=learned-kl-loss/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=11,
            )

        kl_loss_coeff_lr = 1e-5
        cmd(
            "python",
            "src/klpo_stbl_mujoco.py",
            "--seed",
            seed,
            "--env",
            env,
            "--note",
            "learned-kl-loss",
            "--target-kl",
            target_kl,
            "--kl-loss-coeff-lr",
            kl_loss_coeff_lr,
            "--log-dir",
            Out(
                f"klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_kl-loss-coeff={kl_loss_coeff_lr}_note=learned-kl-loss/"
            ),
            warmup_time=3,
            ram_gb=ram_gb,
            priority=20,
        )
        kl_target_stat = "max"
        cmd(
            "python",
            "src/klpo_stbl_mujoco.py",
            "--seed",
            seed,
            "--env",
            env,
            "--note",
            "learned-kl-loss",
            "--target-kl",
            target_kl,
            "--kl-target-stat",
            kl_target_stat,
            "--log-dir",
            Out(
                f"klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_kl-target-stat={kl_target_stat}_note=learned-kl-loss/"
            ),
            warmup_time=3,
            ram_gb=ram_gb,
            priority=21,
        )
        kl_loss_coeff_lr = 1e-2
        for target_kl in [0.1, 0.3, 0.5, 1.0]:
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--note",
                "learned-kl-loss",
                "--target-kl",
                target_kl,
                "--kl-target-stat",
                kl_target_stat,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_kl-target-stat={kl_target_stat}_note=learned-kl-loss/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=22,
            )
        for target_kl in [0.5]:
            for ent_coef in [0.0, 0.01, 0.05, 0.02]:
                for kl_loss_coeff_momentum in [0.0, 0.5, 0.8, 0.9, 0.99]:
                    kl_loss_coeff_lr = 1e-2
                    cmd(
                        "python",
                        "src/klpo_stbl_mujoco.py",
                        "--seed",
                        seed,
                        "--env",
                        env,
                        "--note",
                        "learned-kl-loss",
                        "--target-kl",
                        target_kl,
                        "--kl-target-stat",
                        kl_target_stat,
                        "--kl-loss-coeff-lr",
                        kl_loss_coeff_lr,
                        "--kl-loss-coeff-momentum",
                        kl_loss_coeff_momentum,
                        "--ent-coef",
                        ent_coef,
                        "--log-dir",
                        Out(
                            f"klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_ent-coef={ent_coef}_kl-loss-momentum={kl_loss_coeff_momentum}_note=momentum/"
                        ),
                        warmup_time=3,
                        ram_gb=ram_gb,
                        priority=23,
                    )

# if random.randrange(100) == 0:
# plot_all_csvs.main()
