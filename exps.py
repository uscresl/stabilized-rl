from doexp import cmd, In, Out, GLOBAL_CONTEXT
from socket import gethostname
import random
import json
from subprocess import run

# import plot_all_csvs
import sys
from metaworld.envs.mujoco.env_dict import MT10_V2

# Uncomment to stop starting new runs
# sys.exit(1)

HOST = gethostname()

if HOST == "brain.usc.edu":
    MIN_CONCURRENT_JOBS = 20
    # MIN_CONCURRENT_JOBS = 100
    #GLOBAL_CONTEXT.max_concurrent_jobs = 30
    # GLOBAL_CONTEXT.max_concurrent_jobs = 100
    # GLOBAL_CONTEXT.max_concurrent_jobs = 190
    # GLOBAL_CONTEXT.max_core_alloc = 150
    # GLOBAL_CONTEXT.max_core_alloc = 300
    GLOBAL_CONTEXT.max_core_alloc = 600
    # GLOBAL_CONTEXT.max_concurrent_jobs = 15
    squeue_res = run(['squeue', '--all'], check=False, capture_output=True)
    if squeue_res.returncode == 0:
        out = squeue_res.stdout.decode()
        if "(Priority)" not in out and "(Resources)" not in out:
            # Presumably free resources on the cluster
            GLOBAL_CONTEXT.max_concurrent_jobs += 1
            # print(f"Setting GLOBAL_CONTEXT.max_concurrent_jobs = {GLOBAL_CONTEXT.max_concurrent_jobs}")
        else:
            if GLOBAL_CONTEXT.max_concurrent_jobs != MIN_CONCURRENT_JOBS:
                print(f"GLOBAL_CONTEXT.max_concurrent_jobs was {GLOBAL_CONTEXT.max_concurrent_jobs}")
            # Someone is waiting (maybe us), don't start any more jobs
            GLOBAL_CONTEXT.max_concurrent_jobs = MIN_CONCURRENT_JOBS
        #print(f"Setting GLOBAL_CONTEXT.max_concurrent_jobs = {GLOBAL_CONTEXT.max_concurrent_jobs}")



mujoco_envs = [
    # "InvertedDoublePendulum-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Walker2d-v2",
]
seeds = list(range(16))


ppo_env_names_v3 = [
    "HalfCheetah-v3",
    "Walker2d-v3",
    "Hopper-v3",
    "Swimmer-v3",
    "InvertedPendulum-v2",
    "Reacher-v2",
]

total_steps_for_env = {
    "HalfCheetah-v2": 10_000_000,
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
        ram_gb=6,
        priority=priority,
        cores=cores,
    )


if HOST == "brain.usc.edu":
    for seed in seeds:
        for env, total_steps in [
            ("pick-place-v2", 10_000_000),
            ("window-open-v2", 7_000_000),
            ("button-press-topdown-v2", 7_000_000),
            ("reach-v2", 7_000_000),
            ("push-v2", 7_000_000),
        ]:
            for vf_coef in [
                0.015,
                0.02,
                0.025,
                0.03,
            ]:
                xppo_mt10(
                    seed=seed,
                    priority=(-seed, vf_coef),
                    env=env,
                    note="v_trace_uniform_historical_buffer",
                    add_to_path=[
                        "target_kl",
                        "maximum_kl_loss_coeff",
                        "kl_loss_coeff_lr",
                        "vf_coef",
                        "multi_step_trust_region",
                        "second_loop_vf",
                    ],
                    target_kl=0.02,
                    vf_coef=vf_coef,
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
                    multi_step_trust_region=False,
                    second_loop_vf=True,
                    cores=2,
                )

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
            for kl_target_stat in ["max", "mean"]:
                if kl_target_stat == "max":
                    target_kl_vals = [0.1, 0.2, 0.3, 0.5, 0.7]
                else:
                    target_kl_vals = [0.01, 0.1, 0.15, 0.2, 0.25]
                for target_kl in target_kl_vals:
                    xppo_mujoco(
                        seed=seed,
                        env=env,
                        note="kl_sweep_fixed",
                        target_kl=target_kl,
                        kl_loss_coeff_lr=0.01,
                        kl_target_stat=kl_target_stat,
                    )
            #for kl_target_stat in ["logmax", "cubeispmax"]:
            # for kl_target_stat in ["logmax", "ispmax", "cubeispmax"]:
            #     for kl_loss_coeff_lr in [0.001, 0.01, 0.1, 1.0]:
            #         xppo_mujoco(
            #             seed=seed,
            #             env=env,
            #             note="logmax_sweep_real",
            #             target_kl=0.2,
            #             kl_loss_coeff_lr=kl_loss_coeff_lr,
            #             kl_target_stat=kl_target_stat,
            #         )
            # for kl_loss_coeff_lr in [0.001, 0.01, 0.1, 1.0]:
            #     xppo_mujoco(
            #         seed=seed,
            #         env=env,
            #         note="logmax_sweep",
            #         target_kl=0.2,
            #         kl_loss_coeff_lr=kl_loss_coeff_lr,
            #     )
            # for n_steps in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
            #     historic_buffer_size = n_steps
            #     for second_loop_batch_size in [n_steps, n_steps // 2, n_steps // 4, n_steps // 8, 1024]:
            #         xppo_mujoco(
            #             seed=seed,
            #             env=env,
            #             note="no_historic_buffer_sweep",
            #             target_kl=0.2,
            #             reset_optimizers=False,
            #             kl_loss_coeff_lr=0.1,
            #             n_steps=n_steps,
            #             historic_buffer_size=historic_buffer_size,
            #             second_loop_batch_size=second_loop_batch_size,
            #             priority=(-seed,
            #                       -second_loop_batch_size,
            #                       n_steps),
            #         )

            # for optimize_log_loss_coeff in [False]:
            #     if optimize_log_loss_coeff:
            #         lr_values = [0.05, 0.1, 0.2]
            #     else:
            #         lr_values = [0.01, 0.05, 0.1, 0.15, 0.2]
            #     for kl_loss_coeff_lr in lr_values:
            #         xppo_mujoco(
            #             seed=seed,
            #             env=env,
            #             note="beta_lr_sweep_no_reset_optimizers",
            #             target_kl=0.2,
            #             reset_optimizers=False,
            #             optimize_log_loss_coeff=optimize_log_loss_coeff,
            #             kl_loss_coeff_lr=kl_loss_coeff_lr,
            #         )

            # for n_epochs in [10]:
            #     for kl_loss_coeff_momentum in [0.0, 0.5, 0.99, 'adam']:
            #         for maximum_kl_loss_coeff in [1024]:
            #             for optimize_log_loss_coeff in [False, True]:
            #                 if optimize_log_loss_coeff:
            #                     lr_values = [0.0005, 0.01, 0.05, 0.1]
            #                 else:
            #                     lr_values = [10.0, 20.0, 50.0]
            #                 if kl_loss_coeff_momentum == 'adam':
            #                     kl_loss_coeff_momentum = 0.0
            #                     use_beta_adam = True
            #                 else:
            #                     use_beta_adam = False
            #                 for kl_loss_coeff_lr in lr_values:
            #                     xppo_mujoco(
            #                         seed=seed,
            #                         env=env,
            #                         note="beta_lr_sweep_no_reset2",
            #                         target_kl=0.2,
            #                         reset_beta=False,
            #                         use_beta_adam=use_beta_adam,
            #                         kl_loss_coeff_lr=kl_loss_coeff_lr,
            #                         kl_loss_coeff_momentum=kl_loss_coeff_momentum,
            #                         optimize_log_loss_coeff=optimize_log_loss_coeff,
            #                         maximum_kl_loss_coeff=maximum_kl_loss_coeff,
            #                         n_epochs=n_epochs,
            #                         priority=(maximum_kl_loss_coeff,
            #                                   optimize_log_loss_coeff,
            #                                   -seed,
            #                                   -kl_loss_coeff_momentum,
            #                                   -kl_loss_coeff_lr,
            #                                   )
            #                     )
elif HOST == "resl34":
    # Full GPU utalization
    GLOBAL_CONTEXT.max_concurrent_jobs = 6


    with open('trust-region-layers/configs/pg/mujoco_papi_config.json') as f:
        papi_conf = json.load(f)

    for seed in seeds:
        for env in mujoco_envs:
            conf_name = f"data_tmp/trust-region-layers_papi_seed={seed}_env={env}.json"
            out_dir = f"trust-region-layers_papi_seed={seed}_env={env}/"
            papi_conf["n_envs"] = 1  # Should only affect sampling speed
            papi_conf["seed"] = seed
            papi_conf["env"] = env
            papi_conf["out_dir"] = f"data_tmp/{out_dir}"
            with open(conf_name, 'w') as f:
                json.dump(papi_conf, f, indent=2)
            cmd("python", "trust-region-layers/main.py", conf_name, extra_outputs=[Out(out_dir)],
                cores=3, ram_gb=8)

    # for seed in seeds:
    #     seed = seed + 10000
    #     for env in mujoco_envs:
    #         for n_epochs_args in [{}, {"n_epochs": 20}, {"n_epochs": 30}]:
    #             for optimize_log_loss_coeff in [False, True]:
    #                 if optimize_log_loss_coeff:
    #                     lr_values = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
    #                 else:
    #                     lr_values = [1.0, 5.0, 10.0, 20.0, 50.0]
    #                 for kl_loss_coeff_lr in lr_values:
    #                     xppo_mujoco(
    #                         seed=seed,
    #                         env=env,
    #                         note="beta_lr_sweep_no_reset",
    #                         target_kl=0.2,
    #                         reset_beta=False,
    #                         use_beta_adam=False,
    #                         kl_loss_coeff_lr=kl_loss_coeff_lr,
    #                         kl_loss_coeff_momentum=0.0,
    #                         optimize_log_loss_coeff=optimize_log_loss_coeff,
    #                         maximum_kl_loss_coeff=256,
    #                         **n_epochs_args,
    #                     )
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
            # xppo_mt10(
            #     seed=seed,
            #     priority=(-seed, -maximum_kl_loss_coeff),
            #     env=env,
            #     note="bang_bang_mt10_sweep_fixed_single_step",
            #     add_to_path=[
            #         "target_kl",
            #         "maximum_kl_loss_coeff",
            #         "early_stop_epoch",
            #         "kl_loss_coeff_lr",
            #         "bang_bang_reset_kl_loss_coeff",
            #         "multi_step_trust_region",
            #     ],
            #     target_kl=0.02,
            #     maximum_kl_loss_coeff=maximum_kl_loss_coeff,
            #     kl_target_stat="max",
            #     ent_coef=0.0,
            #     kl_loss_coeff_lr=5.0,
            #     kl_loss_coeff_momentum=0.99999,
            #     historic_buffer_size=48_000,
            #     second_loop_batch_size=24_000,
            #     batch_size=256,
            #     total_steps=total_steps,
            #     bang_bang_kl_loss_opt=True,
            #     bang_bang_reset_kl_loss_coeff=bang_bang_reset_kl_loss_coeff,
            #     early_stop_epoch=early_stop_epoch,
            #     multi_step_trust_region=False,
            # )
            for vf_coef in [
                0.0075,
                0.01,
                0.025,
                0.05,
                # 0.1,
                # 0.2,
                # 0.3,
                # 0.4,
            ]:
                xppo_mt10(
                    seed=seed,
                    priority=(-seed, vf_coef),
                    env=env,
                    note="v_trace_uniform_historical_buffer",
                    add_to_path=[
                        "target_kl",
                        "maximum_kl_loss_coeff",
                        "kl_loss_coeff_lr",
                        "vf_coef",
                        "multi_step_trust_region",
                        "second_loop_vf",
                    ],
                    target_kl=0.02,
                    vf_coef=vf_coef,
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
                    multi_step_trust_region=False,
                    second_loop_vf=True,
                    cores=2,
                )
            # xppo_mt10(
            #     seed=seed,
            #     priority=(-seed, vf_coef),
            #     env=env,
            #     note="no_v_trace_single_step",
            #     add_to_path=[
            #         "target_kl",
            #         "maximum_kl_loss_coeff",
            #         "kl_loss_coeff_lr",
            #         "vf_coef",
            #         "multi_step_trust_region",
            #     ],
            #     target_kl=0.02,
            #     vf_coef=0.1,
            #     kl_target_stat="max",
            #     ent_coef=0.0,
            #     kl_loss_coeff_lr=5.0,
            #     kl_loss_coeff_momentum=0.99999,
            #     historic_buffer_size=48_000,
            #     second_loop_batch_size=24_000,
            #     batch_size=256,
            #     total_steps=total_steps,
            #     bang_bang_kl_loss_opt=False,
            #     v_trace=False,
            #     bang_bang_reset_kl_loss_coeff=False,
            #     early_stop_epoch=False,
            #     multi_step_trust_region=False,
            #     cores=2,
            # )
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
