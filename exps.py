from doexp import cmd, In, Out, GLOBAL_CONTEXT
from socket import gethostname
import random

# import plot_all_csvs
import sys
from metaworld.envs.mujoco.env_dict import MT10_V2

# Uncomment to stop starting new runs
# sys.exit(1)

HOST = gethostname()

if HOST == "brain.usc.edu":
    GLOBAL_CONTEXT.max_concurrent_jobs = 8

mujoco_envs = [
    "InvertedDoublePendulum-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
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
            for target_kl in [0.05, 0.1, 0.3, 0.5, 1.0]:
                for ent_coef in [0.0, 0.01]:
                    for kl_loss_coeff_momentum in [0.99]:
                        for kl_loss_coeff_lr in [1e-2]:
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
                                    f"klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_ent-coef={ent_coef}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-momentum={kl_loss_coeff_momentum}_note=momentum/"
                                ),
                                warmup_time=3,
                                ram_gb=ram_gb,
                                priority=23,
                            )
            target_kl = 0.1
            kl_target_stat = "max"
            kl_loss_coeff_momentum = 0.99
            kl_loss_coeff_lr = 1e-2
            ent_coef = 0.0
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--note",
                "tuned",
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
                    f"klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_ent-coef={ent_coef}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-momentum={kl_loss_coeff_momentum}_note=tuned/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=24,
            )
            for n_steps in [1024, 2048, 4096, 8192]:
                cmd(
                    "python",
                    "src/klpo_stbl_mujoco.py",
                    "--seed",
                    seed,
                    "--env",
                    env,
                    "--note",
                    "batch-size-sweep",
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
                    "--n-steps",
                    n_steps,
                    "--log-dir",
                    Out(
                        f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_note=batch-size-sweep2/"
                    ),
                    warmup_time=3,
                    ram_gb=ram_gb,
                    priority=25
                    + 10
                    * int(
                        env
                        in [
                            "HalfCheetah-v2",
                            "Walker2d-v2",
                        ]
                    )
                    + int(seed / 1000),
                )
            for n_steps in [4096]:
                for kl_loss_coeff_lr in [0.1, 1.0, 2.0, 5.0]:
                    cmd(
                        "python",
                        "src/klpo_stbl_mujoco.py",
                        "--seed",
                        seed,
                        "--env",
                        env,
                        "--note",
                        "batch-size-sweep",
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
                        "--n-steps",
                        n_steps,
                        "--log-dir",
                        Out(
                            f"klpo_stbl/env={env}_seed={seed}_kl-loss-coeff-lr={kl_loss_coeff_lr}_n-steps={n_steps}_note=higher-beta-lr/"
                        ),
                        warmup_time=3,
                        ram_gb=ram_gb,
                        priority=(
                            40,
                            int(env in ["HalfCheetah-v2", "Walker2d-v2"]),
                            seed,
                            -n_steps,
                        ),
                    )
            kl_loss_exp = 1.0
            for n_steps in [4096]:
                for kl_loss_coeff_lr in [1e-3, 1e-2, 0.1, 0.5, 1.0]:
                    cmd(
                        "python",
                        "src/klpo_stbl_mujoco.py",
                        "--seed",
                        seed,
                        "--env",
                        env,
                        "--note",
                        "opt-log-beta2",
                        "--target-kl",
                        target_kl,
                        "--kl-loss-coeff-lr",
                        kl_loss_coeff_lr,
                        "--n-steps",
                        n_steps,
                        "--kl-loss-exp",
                        kl_loss_exp,
                        "--optimize-log-loss-coeff",
                        "--log-dir",
                        Out(
                            f"klpo_stbl/env={env}_seed={seed}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-exp={kl_loss_exp}_n-steps={n_steps}_note=opt-log-beta2/"
                        ),
                        warmup_time=3,
                        ram_gb=ram_gb,
                        priority=(
                            41,
                            int(env in ["HalfCheetah-v2", "Walker2d-v2"]),
                            seed,
                            -n_steps,
                        ),
                    )

else:
    GLOBAL_CONTEXT.max_concurrent_jobs = 3
    ram_gb = 6
    for seed in seeds:
        for env in [
            "pick-place-v2",
            # "window-open-v2",
            # "button-press-topdown-v2",
            # "reach-v2",
        ]:
            for kl_loss_exp in [1, 2]:
                for kl_loss_coeff_lr in [0.01, 0.1, 0.5, 1]:
                    optimize_log_loss_coeff = True
                    target_kl = 0.1
                    kl_target_stat = "max"
                    kl_loss_coeff_momentum = 0.99
                    ent_coef = 0.0
                    cmd(
                        "python",
                        "src/klpo_stbl_MT10.py",
                        "--seed",
                        seed,
                        "--env",
                        env,
                        "--note",
                        "tuned",
                        "--target-kl",
                        target_kl,
                        "--kl-target-stat",
                        kl_target_stat,
                        "--kl-loss-exp",
                        kl_loss_exp,
                        "--optimize-log-loss-coeff",
                        optimize_log_loss_coeff,
                        "--kl-loss-coeff-lr",
                        kl_loss_coeff_lr,
                        "--kl-loss-coeff-momentum",
                        kl_loss_coeff_momentum,
                        "--ent-coef",
                        ent_coef,
                        "--n-steps",
                        4096,
                        "--log-dir",
                        Out(
                            f"MT_10_klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_ent-coef={ent_coef}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-momentum={kl_loss_coeff_momentum}_kl_loss_exp={kl_loss_exp}_optimize_log_loss_coeff={optimize_log_loss_coeff}_note=kl_loss_exp_sweep/"
                        ),
                        warmup_time=3,
                        ram_gb=ram_gb,
                        priority=34,
                    )
        for env in ["reach-v2"]:
            for kl_loss_coeff_lr in [1.0, 2.0]:
                kl_loss_exp = 4
                optimize_log_loss_coeff = False
                target_kl = 0.1
                kl_target_stat = "max"
                kl_loss_coeff_momentum = 0.99
                ent_coef = 0.0
                cmd(
                    "python",
                    "src/klpo_stbl_MT10.py",
                    "--seed",
                    seed,
                    "--env",
                    env,
                    "--note",
                    "tuned",
                    "--target-kl",
                    target_kl,
                    "--kl-target-stat",
                    kl_target_stat,
                    "--kl-loss-exp",
                    kl_loss_exp,
                    "--optimize-log-loss-coeff",
                    optimize_log_loss_coeff,
                    "--kl-loss-coeff-lr",
                    kl_loss_coeff_lr,
                    "--kl-loss-coeff-momentum",
                    kl_loss_coeff_momentum,
                    "--ent-coef",
                    ent_coef,
                    "--n-steps",
                    4096,
                    "--log-dir",
                    Out(
                        f"MT_10_klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_ent-coef={ent_coef}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-momentum={kl_loss_coeff_momentum}_kl_loss_exp={kl_loss_exp}_optimize_log_loss_coeff={optimize_log_loss_coeff}_note=kl_loss_coeff_sweep/"
                    ),
                    warmup_time=3,
                    ram_gb=ram_gb,
                    priority=28,
                )
            for kl_loss_coeff_lr in [0.05, 0.1, 0.5, 1.0, 2.0]:
                kl_loss_exp = 4
                optimize_log_loss_coeff = True
                target_kl = 0.1
                kl_target_stat = "max"
                kl_loss_coeff_momentum = 0.99
                ent_coef = 0.0
                cmd(
                    "python",
                    "src/klpo_stbl_MT10.py",
                    "--seed",
                    seed,
                    "--env",
                    env,
                    "--note",
                    "tuned",
                    "--target-kl",
                    target_kl,
                    "--kl-target-stat",
                    kl_target_stat,
                    "--kl-loss-exp",
                    kl_loss_exp,
                    "--optimize-log-loss-coeff",
                    optimize_log_loss_coeff,
                    "--kl-loss-coeff-lr",
                    kl_loss_coeff_lr,
                    "--kl-loss-coeff-momentum",
                    kl_loss_coeff_momentum,
                    "--ent-coef",
                    ent_coef,
                    "--n-steps",
                    4096,
                    "--log-dir",
                    Out(
                        f"MT_10_klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_ent-coef={ent_coef}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-momentum={kl_loss_coeff_momentum}_kl_loss_exp={kl_loss_exp}_optimize_log_loss_coeff={optimize_log_loss_coeff}_note=kl_loss_coeff_sweep/"
                    ),
                    warmup_time=3,
                    ram_gb=ram_gb,
                    priority=28,
                )

            for optimize_log_loss_coeff in [True, False]:
                kl_loss_exp = 4
                kl_loss_coeff_lr = 0.5
                target_kl = 0.1
                kl_target_stat = "max"
                kl_loss_coeff_momentum = 0.99
                ent_coef = 0.0
                cmd(
                    "python",
                    "src/klpo_stbl_MT10.py",
                    "--seed",
                    seed,
                    "--env",
                    env,
                    "--note",
                    "tuned",
                    "--target-kl",
                    target_kl,
                    "--kl-target-stat",
                    kl_target_stat,
                    "--kl-loss-exp",
                    kl_loss_exp,
                    "--optimize-log-loss-coeff",
                    optimize_log_loss_coeff,
                    "--kl-loss-coeff-lr",
                    kl_loss_coeff_lr,
                    "--kl-loss-coeff-momentum",
                    kl_loss_coeff_momentum,
                    "--ent-coef",
                    ent_coef,
                    "--n-steps",
                    4096,
                    "--log-dir",
                    Out(
                        f"MT_10_klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_ent-coef={ent_coef}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-momentum={kl_loss_coeff_momentum}_kl_loss_exp={kl_loss_exp}_optimize_log_loss_coeff={optimize_log_loss_coeff}_note=log_loss_sweep/"
                    ),
                    warmup_time=3,
                    ram_gb=ram_gb,
                    priority=26,
                )

# if random.randrange(100) == 0:
#     plot_all_csvs.main()
