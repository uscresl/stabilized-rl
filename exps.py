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
    GLOBAL_CONTEXT.max_concurrent_jobs = 8
    for seed in seeds:
        target_kl = 0.2
        ram_gb = 4
        for env in mujoco_envs:
            target_kl = 0.03
            kl_loss_coeff_lr = 0.1
            kl_loss_coeff_momentum = 0.9999
            n_steps = 4096
            note = "minibatch-kl4"
            for minibatch_kl_penalty in [True, False]:
                cmd(
                    "python",
                    "src/klpo_stbl_mujoco.py",
                    "--seed",
                    seed,
                    "--env",
                    env,
                    "--target-kl",
                    target_kl,
                    "--kl-loss-coeff-lr",
                    kl_loss_coeff_lr,
                    "--kl-loss-coeff-momentum",
                    kl_loss_coeff_momentum,
                    "--n-steps",
                    n_steps,
                    "--note",
                    note,
                    "--minibatch-kl-penalty=yes"
                    if minibatch_kl_penalty
                    else "--minibatch-kl-penalty=no",
                    "--log-dir",
                    Out(
                        f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_minibatch-kl-penalty={minibatch_kl_penalty}_note={note}/"
                    ),
                    warmup_time=3,
                    ram_gb=ram_gb,
                    priority=(
                        51,
                        minibatch_kl_penalty,
                        int(env in ["HalfCheetah-v2", "Walker2d-v2"]),
                        seed,
                    ),
                )
            note = "beta-adam-opt"
            for kl_loss_coeff_lr in [5.0, 10.0]:
                cmd(
                    "python",
                    "src/klpo_stbl_mujoco.py",
                    "--seed",
                    seed,
                    "--env",
                    env,
                    "--target-kl",
                    target_kl,
                    "--kl-loss-coeff-lr",
                    kl_loss_coeff_lr,
                    "--kl-loss-coeff-momentum",
                    kl_loss_coeff_momentum,
                    "--n-steps",
                    n_steps,
                    "--note",
                    note,
                    "--minibatch-kl-penalty=yes",
                    "--use-beta-adam=yes",
                    "--log-dir",
                    Out(
                        f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_beta-adam-opt=True_note={note}/"
                    ),
                    warmup_time=3,
                    ram_gb=ram_gb,
                    priority=(
                        51 + 2 * int(env in ["HalfCheetah-v2"] and kl_loss_coeff_lr == 5.0),
                        seed,
                        int(env in ["HalfCheetah-v2"]),
                    ),
                )
            note = "sparse-second-loop2"
            kl_loss_coeff_lr = 1.0
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--kl-loss-coeff-momentum",
                kl_loss_coeff_momentum,
                "--n-steps",
                n_steps,
                "--note",
                note,
                "--minibatch-kl-penalty=yes",
                "--sparse-second-loop=yes",
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    50,
                    seed,
                    int(env in ["HalfCheetah-v2"]),
                    int(env in ["HalfCheetah-v2", "Walker2d-v2"]),
                ),
            )
            note = "no-normalize-advantages"
            cmd(
                "python",
                "src/klpo_stbl_mujoco.py",
                "--seed",
                seed,
                "--env",
                env,
                "--target-kl",
                target_kl,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--kl-loss-coeff-momentum",
                kl_loss_coeff_momentum,
                "--n-steps",
                n_steps,
                "--note",
                note,
                "--minibatch-kl-penalty=yes",
                "--sparse-second-loop=yes",
                "--normalize-advantage=no",
                "--log-dir",
                Out(
                    f"klpo_stbl/env={env}_seed={seed}_n-steps={n_steps}_target-kl={target_kl}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(
                    52,
                    int(env in ["HalfCheetah-v2"]),
                    seed,
                ),
            )

elif HOST == "stygian":
    GLOBAL_CONTEXT.max_concurrent_jobs = 3
    ram_gb = 9
    # for seed in seeds:
    #     for env in [
    #         "pick-place-v2",
    #         # "window-open-v2",
    #         # "button-press-topdown-v2",
    #         # "reach-v2",
    #     ]:
    #         total_steps: int = 20_000_000
    #         n_steps = 50_000
    #         gamma = 0.99
    #         batch_size = 32
    #         gae_lambda = 0.95
    #         learning_rate = 5e-4
    #         n_epochs = 10

    #         note = "basline_ppo"
    #         cmd(
    #             "python",
    #             "src/ppo_stbl_MT10.py",
    #             "--seed",
    #             seed,
    #             "--env",
    #             env,
    #             "--total-steps",
    #             total_steps,
    #             "--n-steps",
    #             n_steps,
    #             "--gamma",
    #             gamma,
    #             "--batch-size",
    #             batch_size,
    #             "--gae-lambda",
    #             gae_lambda,
    #             "--learning-rate",
    #             learning_rate,
    #             "--n-epochs",
    #             n_epochs,
    #             "--note",
    #             note,
    #             "--log-dir",
    #             Out(f"PPO_stbl_MT10_baseline/env={env}_seed={seed}_note={note}/"),
    #             warmup_time=3,
    #             ram_gb=ram_gb,
    #             priority=(35, -seed),
    #         )
    for seed in seeds:
        for env in [
            "pick-place-v2",
            # "window-open-v2",
            # "button-press-topdown-v2",
            # "reach-v2",
        ]:
            total_steps: int = 20_000_000
            batch_size = 50_000
            center_adv = True
            normalize_env = True
            note = "basline_garage_ppo"
            cmd(
                "python",
                "src/ppo_MT10.py",
                "--seed",
                seed,
                "--env",
                env,
                "--batch-size",
                batch_size,
                "--total-steps",
                total_steps,
                "--log-dir",
                Out(f"PPO_garage_MT10_baseline/env={env}_seed={seed}_note={note}/"),
                "--note",
                note,
                "--center-adv",
                center_adv,
                "--normalize-env",
                normalize_env,
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(-seed, 35),
            )
    for seed in seeds:
        for env in [
            "pick-place-v2",
            # "window-open-v2",
            # "button-press-topdown-v2",
            # "reach-v2",
        ]:
            note = "tuned_xppo"
            optimize_log_loss_coeff = False
            second_penalty_loop = True
            reset_policy_optimizer = True

            kl_target_stat = "max"
            ent_coef = 0.0
            target_kl = 1.5e-3
            kl_loss_coeff_lr = 3.0
            kl_loss_coeff_momentum = 0.99999
            historic_buffer_size = 32_000
            cmd(
                "python",
                "src/klpo_stbl_MT10.py",
                "--seed",
                seed,
                "--env",
                env,
                "--target-kl",
                target_kl,
                "--kl-target-stat",
                kl_target_stat,
                "--optimize-log-loss-coeff",
                optimize_log_loss_coeff,
                "--kl-loss-coeff-lr",
                kl_loss_coeff_lr,
                "--kl-loss-coeff-momentum",
                kl_loss_coeff_momentum,
                "--second-penalty-loop",
                second_penalty_loop,
                "--reset-policy-optimizer",
                reset_policy_optimizer,
                "--ent-coef",
                ent_coef,
                "--n-steps",
                4096,
                "--total-steps",
                20_000_000,
                "--historic-buffer-size",
                historic_buffer_size,
                "--note",
                note,
                "--log-dir",
                Out(
                    f"MT_10_klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-coeff-momentum={kl_loss_coeff_momentum}_historic-buffer-size={historic_buffer_size}_note={note}/"
                ),
                warmup_time=3,
                ram_gb=ram_gb,
                priority=(-seed, 36),
            )
# if random.randrange(100) == 0:
#     plot_all_csvs.main()
