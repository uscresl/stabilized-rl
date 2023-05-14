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
    # "InvertedDoublePendulum-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Walker2d-v2",
]
seeds = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]


ppo_env_names_v3 = [
    "HalfCheetah-v3",
    "Walker2d-v3",
    "Hopper-v3",
    "Swimmer-v3",
    "InvertedPendulum-v2",
    "Reacher-v2",
]

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
elif HOST == "resl34":
    GLOBAL_CONTEXT.max_concurrent_jobs = 4
    for seed in seeds:
        ram_gb = 4
        for env in mujoco_envs:
            total_steps = 20_000_000
            note = "baseline_ppo_10m"
            cmd(
                "python",
                "src/ppo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--note",
                note,
                "--log-dir",
                Out(
                    f"ppo_stbl/env={env}_seed={seed}_note={note}/"
                ),
                priority=(
                    64,
                    int(env in ["Walker2d-v2"]),
                    int(env in ["Hopper-v2"]),
                    seed,
                ),
            )
            batch_size_args = ()
            batch_size_postfix = "10m-512-5"
            batch_size = 512
            target_kl = 0.03
            kl_loss_coeff_lr = 5.0
            n_steps = 4096
            note = "xppo" + batch_size_postfix
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--n-steps",
                n_steps,
                *batch_size_args,
                "--note",
                note,
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    64,
                    int(env in ["Walker2d-v2"]),
                    int(env in ["Hopper-v2"]),
                    seed,
                ),
            )
            note = "one-phase" + batch_size_postfix
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--n-steps",
                n_steps,
                *batch_size_args,
                "--note",
                note,
                "--second-penalty-loop=no",
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    60,
                    int(env in ["HalfCheetah-v2"]),
                    seed,
                ),
            )
            note = "mean-kl-target" + batch_size_postfix
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--n-steps",
                n_steps,
                *batch_size_args,
                "--note",
                note,
                "--kl-target-stat=mean",
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    60,
                    int(env in ["HalfCheetah-v2"]),
                    seed,
                ),
            )
            note = "small-mean-kl-target" + batch_size_postfix
            target_kl = 0.003
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--n-steps",
                n_steps,
                *batch_size_args,
                "--note",
                note,
                "--kl-target-stat=mean",
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    61,
                    int(env in ["HalfCheetah-v2"]),
                    seed,
                ),
            )
            target_kl = 0.03
            note = "no-reset" + batch_size_postfix
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--n-steps",
                n_steps,
                *batch_size_args,
                "--note",
                note,
                "--reset-optimizers=no",
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    60,
                    int(env in ["HalfCheetah-v2"]),
                    seed,
                ),
            )
            note = "no-historic" + batch_size_postfix
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--n-steps",
                n_steps,
                *batch_size_args,
                "--note",
                note,
                "--historic-buffer-size=4096",
                "--second-loop-batch-size=4096",
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    60,
                    int(env in ["HalfCheetah-v2"]),
                    -batch_size,
                    seed,
                ),
            )
            note = "second-loop-vf" + batch_size_postfix
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--n-steps",
                n_steps,
                *batch_size_args,
                "--note",
                note,
                "--second-loop-vf=yes",
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    61,
                    int(env in ["HalfCheetah-v2"]),
                    -batch_size,
                    seed,
                ),
            )
            # note = "no-reset-log-beta" + batch_size_postfix
            # cmd(
            #     "python",
            #     "src/klpo_stbl_mujoco.py",
            #     "--seed",
            #     seed,
            #     "--env",
            #     env,
            #     "--target-kl",
            #     target_kl,
            #     "--kl-loss-coeff-lr",
            #     kl_loss_coeff_lr,
            #     "--n-steps",
            #     n_steps,
            #     *batch_size_args,
            #     "--note",
            #     note,
            #     "--reset-optimizers=no",
            #     "--optimize-log-loss-coeff=yes",
            #     "--log-dir",
            #     Out(
            #         f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
            #     ),
            #     warmup_time=3,
            #     ram_gb=ram_gb,
            #     priority=(
            #         60,
            #         int(env in ["HalfCheetah-v2"]),
            #         seed,
            #     ),
            # )
            total_steps = 20_000_000
            note = "small-log-beta" + batch_size_postfix
            for kl_loss_coeff_lr in (0.1, 1.0):
                cmd(
                    "python",
                    "src/klpo_stbl_mujoco.py",
                    "--seed",
                    seed,
                    "--env",
                    env,
                    "--total-steps",
                    total_steps,
                    "--target-kl",
                    target_kl,
                    "--kl-loss-coeff-lr",
                    kl_loss_coeff_lr,
                    *batch_size_args,
                    "--note",
                    note,
                    "--optimize-log-loss-coeff=yes",
                    "--log-dir",
                    Out(
                        f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_kl-loss-coeff-lr={kl_loss_coeff_lr}_note={note}/"
                    ),
                    warmup_time=3,
                    ram_gb=ram_gb,
                    priority=(
                        59,
                        int(env in ["Walker2d-v2"]) + int(kl_loss_coeff_lr == 1.0),
                        seed,
                    ),
                )

elif HOST == "stygian":
    GLOBAL_CONTEXT.max_concurrent_jobs = 3
    ram_gb = 9
    for seed in seeds:
        for env in [
            "pick-place-v2",
            # "window-open-v2",
            # "button-press-topdown-v2",
            # "reach-v2",
        ]:
            total_steps: int = 20_000_000
            n_steps = 2048
            gamma = 0.99
            batch_size = 64
            gae_lambda = 0.95
            learning_rate = 5e-4
            n_epochs = 10

            note = "basline_stbl_ppo"
            cmd(
                "python",
                "src/ppo_stbl_MT10.py",
                "--seed",
                seed,
                "--env",
                env,
                "--total-steps",
                total_steps,
                "--n-steps",
                n_steps,
                "--gamma",
                gamma,
                "--batch-size",
                batch_size,
                "--gae-lambda",
                gae_lambda,
                "--learning-rate",
                learning_rate,
                "--n-epochs",
                n_epochs,
                "--note",
                note,
                "--log-dir",
                Out(f"PPO_stbl_MT10_baseline/env={env}_seed={seed}_note={note}/"),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(35, -seed),
            )
    # for seed in seeds:
    #     for env, total_steps in [
    #         ("pick-place-v2", 20_000_000),
    #         # ("window-open-v2", 7_000_000),
    #         # ("button-press-topdown-v2", 7_000_000),
    #         # ("reach-v2", 7_000_000),
    #     ]:
    #         batch_size = 50_000
    #         center_adv = False
    #         normalize_env = True
    #         gae_lambda = 0.95
    #         note = "basline_garage_max_entropy_ppo"
    #         entropy_method = "max"
    #         policy_ent_coeff = 0.01
    #         stop_entropy_gradient = True
    #         cmd(
    #             "python",
    #             "src/ppo_MT10.py",
    #             "--seed",
    #             seed,
    #             "--env",
    #             env,
    #             "--batch-size",
    #             batch_size,
    #             "--total-steps",
    #             total_steps,
    #             "--entropy-method",
    #             entropy_method,
    #             "--policy-ent-coeff",
    #             policy_ent_coeff,
    #             "--stop-entropy-gradient",
    #             stop_entropy_gradient,
    #             "--gae-lambda",
    #             gae_lambda,
    #             "--log-dir",
    #             Out(f"PPO_garage_MT10_baseline/env={env}_seed={seed}_note={note}/"),
    #             "--note",
    #             note,
    #             "--center-adv",
    #             center_adv,
    #             "--normalize-env",
    #             normalize_env,
    #             warmup_time=3,
    #             ram_gb=ram_gb,
    #             priority=(-seed, 37),
    #         )
    # for seed in seeds:
    #     for env, total_steps, target_kl in [
    #         # ("pick-place-v2", 20_000_000),
    #         # ("window-open-v2", 7_000_000),
    #         # ("button-press-topdown-v2", 7_000_000),
    #         # ("reach-v2", 7_000_000),
    #         ("push-v2", 20_000_000, 0.75e-3),
    #     ]:
    #         note = "tuned_xppo"
    #         optimize_log_loss_coeff = False
    #         second_penalty_loop = True
    #         reset_optimizers = True

    #         kl_target_stat = "max"
    #         ent_coef = 0.0
    #         kl_loss_coeff_lr = 5.0
    #         kl_loss_coeff_momentum = 0.99999
    #         historic_buffer_size = 32_000
    #         second_loop_batch_size = 16_000
    #         batch_size = 256
    #         cmd(
    #             "python",
    #             "src/klpo_stbl_MT10.py",
    #             "--seed",
    #             seed,
    #             "--env",
    #             env,
    #             "--target-kl",
    #             target_kl,
    #             "--kl-target-stat",
    #             kl_target_stat,
    #             "--optimize-log-loss-coeff",
    #             optimize_log_loss_coeff,
    #             "--kl-loss-coeff-lr",
    #             kl_loss_coeff_lr,
    #             "--kl-loss-coeff-momentum",
    #             kl_loss_coeff_momentum,
    #             "--second-penalty-loop",
    #             second_penalty_loop,
    #             "--reset-optimizers",
    #             reset_optimizers,
    #             "--ent-coef",
    #             ent_coef,
    #             "--second-loop-batch-size",
    #             second_loop_batch_size,
    #             "--batch-size",
    #             batch_size,
    #             "--n-steps",
    #             4096,
    #             "--total-steps",
    #             total_steps,
    #             "--historic-buffer-size",
    #             historic_buffer_size,
    #             "--note",
    #             note,
    #             "--log-dir",
    #             Out(
    #                 f"MT_10_klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-coeff-momentum={kl_loss_coeff_momentum}_historic-buffer-size={historic_buffer_size}_note={note}/"
    #             ),
    #             warmup_time=3,
    #             ram_gb=ram_gb,
    #             priority=(-seed, 36),
    #         )
    # for seed in seeds[:4]:
    #     for env, total_steps in [
    #         ("reach-v2", 7_000_000),
    #         ("push-v2", 7_000_000),
    #         ("door-open-v2", 7_000_000),
    #         ("drawer-open-v2", 7_000_000),
    #         ("drawer-close-v2", 7_000_000),
    #         ("button-press-topdown-v2", 7_000_000),
    #         ("peg-insert-side-v2", 7_000_000),
    #         # ("window-open-v2", 7_000_000),
    #         ("window-close-v2", 7_000_000),
    #         ("pick-place-v2", 7_000_000),
    #     ]:
    #         note = "xppo_transfer_exp_window_open"
    #         optimize_log_loss_coeff = False
    #         second_penalty_loop = True
    #         reset_optimizers = True

    #         kl_target_stat = "max"
    #         ent_coef = 0.0
    #         target_kl = 1.5e-3
    #         kl_loss_coeff_lr = 3.0
    #         kl_loss_coeff_momentum = 0.99999
    #         historic_buffer_size = 32_000
    #         cmd(
    #             "python",
    #             "src/xppo_transfer_MT10.py",
    #             "--seed",
    #             seed,
    #             "--env",
    #             env,
    #             "--target-kl",
    #             target_kl,
    #             "--kl-target-stat",
    #             kl_target_stat,
    #             "--optimize-log-loss-coeff",
    #             optimize_log_loss_coeff,
    #             "--kl-loss-coeff-lr",
    #             kl_loss_coeff_lr,
    #             "--kl-loss-coeff-momentum",
    #             kl_loss_coeff_momentum,
    #             "--second-penalty-loop",
    #             second_penalty_loop,
    #             "--reset-optimizers",
    #             reset_optimizers,
    #             "--ent-coef",
    #             ent_coef,
    #             "--n-steps",
    #             4096,
    #             "--total-steps",
    #             total_steps,
    #             "--historic-buffer-size",
    #             historic_buffer_size,
    #             "--note",
    #             note,
    #             "--log-dir",
    #             Out(
    #                 f"MT_10_transfer_exp/env={env}_seed={seed}_target-kl={target_kl}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-coeff-momentum={kl_loss_coeff_momentum}_historic-buffer-size={historic_buffer_size}_note={note}/"
    #             ),
    #             warmup_time=3,
    #             ram_gb=ram_gb,
    #             priority=(-seed, 36),
    #         )
# if random.randrange(100) == 0:
#     plot_all_csvs.main()
