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
    # GLOBAL_CONTEXT.max_concurrent_jobs = 190
    # GLOBAL_CONTEXT.max_concurrent_jobs = 50
    GLOBAL_CONTEXT.max_concurrent_jobs = 100
    # GLOBAL_CONTEXT.max_core_alloc = 432
    # GLOBAL_CONTEXT.max_core_alloc = 150
    GLOBAL_CONTEXT.max_core_alloc = 300

mujoco_envs = [
    # "InvertedDoublePendulum-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Walker2d-v2",
]
seeds = [
    1111,
    2222,
    3333,
    4444,
    5555,
    6666,
    7777,
    8888,
    # 1, 2, 3, 4, 5, 6, 7, 8
]


ppo_env_names_v3 = [
    "HalfCheetah-v3",
    "Walker2d-v3",
    "Hopper-v3",
    "Swimmer-v3",
    "InvertedPendulum-v2",
    "Reacher-v2",
]

total_steps_for_env = {
    "HalfCheetah-v2": 3_000_000,
    "Hopper-v2": 10_000_000,
    "Walker2d-v2": 10_000_000,
}


def xppo_mujoco(
    seed, env, note, priority=None, cores=3, add_to_path: list = None, **kwargs
):
    total_steps = total_steps_for_env.get(env, 20_000_000)
    if priority is None:
        priority = (50, total_steps, -seed)
    if add_to_path is None:
        add_to_path = [k for k, _ in kwargs.items()][:5]
    kwargs_path = "_".join(
        f"{k.replace('_', '-')}={kwargs.get(k)}" for k in add_to_path
    )
    return cmd(
        "python",
        "src/klpo_stbl_mujoco.py",
        "--seed",
        seed,
        "--env",
        env,
        "--total-steps",
        total_steps,
        *[f"--{k.replace('_', '-')}={v}" for (k, v) in kwargs.items()],
        "--note",
        note,
        "--log-dir",
        Out(f"klpo_stbl/env={env}_seed={seed}_{kwargs_path}_note={note}/"),
        warmup_time=3,
        ram_gb=6,
        priority=priority,
        cores=cores,
    )


def xppo_mt10(
    seed, env, note, priority=None, cores=3, add_to_path: list = None, **kwargs
):
    if kwargs.get("total_steps"):
        total_steps = kwargs.pop("total_steps")
    else:
        total_steps = total_steps_for_env.get(env, 20_000_000)

    if priority is None:
        priority = (50, total_steps, -seed)
    if add_to_path is None:
        add_to_path = [k for k, _ in kwargs.items()][:5]
    kwargs_path = "_".join(
        f"{k.replace('_', '-')}={kwargs.get(k)}" for k in add_to_path
    )
    return cmd(
        "python",
        "src/klpo_stbl_MT10.py",
        "--seed",
        seed,
        "--env",
        env,
        "--total-steps",
        total_steps,
        *[f"--{k.replace('_', '-')}={v}" for (k, v) in kwargs.items()],
        "--note",
        note,
        "--log-dir",
        Out(f"klpo_stbl/env={env}_seed={seed}_{kwargs_path}_note={note}/"),
        warmup_time=3,
        ram_gb=8,
        priority=priority,
        cores=cores,
    )


if HOST == "brain.usc.edu":
    for seed in seeds:
        for env in [
            "pick-place-v2",
            "window-open-v2",
            # "button-press-topdown-v2",
            "reach-v2",
        ]:
            pass

    for seed in seeds:
        for env in mujoco_envs:
            for optimize_log_loss_coeff in [False, True]:
                if optimize_log_loss_coeff:
                    lr_values = [0.01, 0.1, 1.0]
                else:
                    lr_values = [1.0, 5.0, 10.0, 20.0]
                for kl_loss_coeff_lr in lr_values:
                    xppo_mujoco(
                        seed=seed,
                        env=env,
                        note="beta_lr_sweep_no_reset",
                        target_kl=0.2,
                        reset_beta=False,
                        use_beta_adam=False,
                        kl_loss_coeff_lr=kl_loss_coeff_lr,
                        kl_loss_coeff_momentum=0.0,
                        optimize_log_loss_coeff=optimize_log_loss_coeff,
                        maximum_kl_loss_coeff=256,
                    )
elif HOST == "resl34":
    # Full GPU utalization
    GLOBAL_CONTEXT.max_concurrent_jobs = 6

    for seed in seeds:
        seed = seed + 10000
        for env in mujoco_envs:
            for optimize_log_loss_coeff in [False, True]:
                if optimize_log_loss_coeff:
                    lr_values = [0.01, 0.1, 1.0]
                else:
                    lr_values = [1.0, 5.0, 10.0, 20.0]
                for kl_loss_coeff_lr in lr_values:
                    xppo_mujoco(
                        seed=seed,
                        env=env,
                        note="beta_lr_sweep_no_reset",
                        target_kl=0.2,
                        reset_beta=False,
                        use_beta_adam=False,
                        kl_loss_coeff_lr=kl_loss_coeff_lr,
                        kl_loss_coeff_momentum=0.0,
                        optimize_log_loss_coeff=optimize_log_loss_coeff,
                        maximum_kl_loss_coeff=256,
                    )
elif HOST == "stygian":
    GLOBAL_CONTEXT.max_concurrent_jobs = 4
    ram_gb = 4
    early_stop_epoch = False

    for seed in seeds[:5]:
        for env, total_steps in [
            ("pick-place-v2", 10_000_000),
            ("window-open-v2", 7_000_000),
            ("button-press-topdown-v2", 7_000_000),
            ("reach-v2", 7_000_000),
            ("push-v2", 7_000_000),
        ]:
            # for maximum_kl_loss_coeff in [10, 50, 75, 100]:
            #     for early_stop_epoch, bang_bang_reset_kl_loss_coeff in [
            #         (False, False),
            #     ]:
            #         xppo_mt10(
            #             seed=seed,
            #             priority=(-seed, -maximum_kl_loss_coeff),
            #             env=env,
            #             note="bang_bang_mt10_sweep_fixed",
            #             add_to_path=[
            #                 "target_kl",
            #                 "maximum_kl_loss_coeff",
            #                 "early_stop_epoch",
            #                 "kl_loss_coeff_lr",
            #                 "bang_bang_reset_kl_loss_coeff",
            #             ],
            #             target_kl=0.02,
            #             maximum_kl_loss_coeff=maximum_kl_loss_coeff,
            #             kl_target_stat="max",
            #             ent_coef=0.0,
            #             kl_loss_coeff_lr=5.0,
            #             kl_loss_coeff_momentum=0.99999,
            #             historic_buffer_size=48_000,
            #             second_loop_batch_size=24_000,
            #             batch_size=256,
            #             total_steps=total_steps,
            #             bang_bang_kl_loss_opt=True,
            #             bang_bang_reset_kl_loss_coeff=bang_bang_reset_kl_loss_coeff,
            #             early_stop_epoch=early_stop_epoch,
            #         )
            xppo_mt10(
                seed=seed,
                priority=(-seed),
                env=env,
                note="v_trace",
                add_to_path=[
                    "target_kl",
                    "maximum_kl_loss_coeff",
                    "early_stop_epoch",
                    "kl_loss_coeff_lr",
                    "bang_bang_reset_kl_loss_coeff",
                ],
                target_kl=0.02,
                kl_target_stat="max",
                ent_coef=0.0,
                kl_loss_coeff_lr=5.0,
                kl_loss_coeff_momentum=0.99999,
                historic_buffer_size=48_000,
                second_loop_batch_size=24_000,
                batch_size=256,
                total_steps=total_steps,
                bang_bang_kl_loss_opt=False,
                v_trace=True,
                bang_bang_reset_kl_loss_coeff=False,
                early_stop_epoch=False,
            )
    # for seed in seeds:
    #     for env in [
    #         # "pick-place-v2",
    #         # "window-open-v2",
    #         # "button-press-topdown-v2",
    #         # "reach-v2",
    #     ]:
    #         total_steps: int = 20_000_000
    #         n_steps = 2048
    #         gamma = 0.99
    #         batch_size = 64
    #         gae_lambda = 0.95
    #         learning_rate = 5e-4
    #         n_epochs = 10

    #         note = "basline_stbl_ppo"
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
    #     for env, total_steps in [
    #         ("pick-place-v2", 20_000_000),
    #         # ("window-open-v2", 7_000_000),
    #         # ("button-press-topdown-v2", 7_000_000),
    #         # ("reach-v2", 7_000_000),
    #         # ("push-v2", 7_000_000),
    #     ]:
    #         target_kl = 0.02
    #         note = "xppo_early_stop_within_epoch"
    #         optimize_log_loss_coeff = False
    #         second_penalty_loop = True
    #         reset_optimizers = True

    #         kl_target_stat = "max"
    #         ent_coef = 0.0
    #         kl_loss_coeff_lr = 5.0
    #         kl_loss_coeff_momentum = 0.99999
    #         historic_buffer_size = 48_000
    #         second_loop_batch_size = 24_000
    #         batch_size = 256
    #         early_stop_epoch = True
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
    #             "--optimize-log-loss-coeff=no",
    #             "--multi-step-trust-region=no",
    #             "--second-penalty-loop",
    #             "--reset-optimizers",
    #             "--early-stop-epoch",
    #             "--ent-coef",
    #             ent_coef,
    #             "--kl-loss-coeff-lr",
    #             kl_loss_coeff_lr,
    #             "--kl-loss-coeff-momentum",
    #             kl_loss_coeff_momentum,
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
    #                 f"MT_10_klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-coeff-momentum={kl_loss_coeff_momentum}_early-stop-epoch={early_stop_epoch}_note={note}/"
    #             ),
    #             warmup_time=3,
    #             ram_gb=ram_gb,
    #             priority=(-seed, 36),
    #         )
    # for seed in seeds:
    #     for env, total_steps in [
    #         ("pick-place-v2", 20_000_000),
    #         # ("window-open-v2", 7_000_000),
    #         # ("button-press-topdown-v2", 7_000_000),
    #         # ("reach-v2", 7_000_000),
    #         # ("push-v2", 7_000_000),
    #     ]:
    #         target_kl = 0.02
    #         note = "xppo_early_stop_across_epochs"
    #         optimize_log_loss_coeff = False
    #         second_penalty_loop = True
    #         reset_optimizers = True

    #         kl_target_stat = "max"
    #         ent_coef = 0.0
    #         kl_loss_coeff_lr = 5.0
    #         kl_loss_coeff_momentum = 0.99999
    #         historic_buffer_size = 48_000
    #         second_loop_batch_size = 24_000
    #         batch_size = 256
    #         early_stop_across_epochs = True
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
    #             "--optimize-log-loss-coeff=no",
    #             "--multi-step-trust-region=no",
    #             "--second-penalty-loop",
    #             "--reset-optimizers",
    #             "--early-stop-across-epochs",
    #             "--ent-coef",
    #             ent_coef,
    #             "--kl-loss-coeff-lr",
    #             kl_loss_coeff_lr,
    #             "--kl-loss-coeff-momentum",
    #             kl_loss_coeff_momentum,
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
    #                 f"MT_10_klpo_stbl/env={env}_seed={seed}_target-kl={target_kl}_kl-loss-coeff-lr={kl_loss_coeff_lr}_kl-loss-coeff-momentum={kl_loss_coeff_momentum}_early-stop-across-epochs={early_stop_across_epochs}_note={note}/"
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
    #         ("window-open-v2", 7_000_000),
    #         ("window-close-v2", 7_000_000),
    #         # ("pick-place-v2", 7_000_000),
    #     ]:
    #         note = "xppo_transfer_exp_pick-place"
    #         optimize_log_loss_coeff = False
    #         second_penalty_loop = True
    #         reset_optimizers = True
    #         kl_target_stat = "max"
    #         ent_coef = 0.0
    #         target_kl = 0.02
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
    #             "--multi-step-trust-region=no",
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
