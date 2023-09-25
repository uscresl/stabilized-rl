from doexp import cmd, In, Out, GLOBAL_CONTEXT
from socket import gethostname
from subprocess import run
import math


HOST = gethostname()

# XXX Remember to redact these before review
WANDB_ENTITY = "resl-mixppo"
BRAIN_HOSTNAME = "brain.usc.edu"

if HOST == BRAIN_HOSTNAME:
    MIN_CONCURRENT_JOBS = 10
    GLOBAL_CONTEXT.max_core_alloc = 600
    squeue_res = run(["squeue", "--all"], check=False, capture_output=True)
    if squeue_res.returncode == 0:
        out = squeue_res.stdout.decode()
        found_waiting_normal_job = False
        for line in out.split('\n'):
            if "(Priority)" in line or "(Resources)" in line:
                if "slurm-lon" not in line:
                    found_waiting_normal_job = True
                    # print("Found waiting job")
                    # print(line)
        if not found_waiting_normal_job:
            # Presumably free resources on the cluster
            GLOBAL_CONTEXT.max_concurrent_jobs += 1
        else:
            if GLOBAL_CONTEXT.max_concurrent_jobs > len(GLOBAL_CONTEXT.running):
                # print(
                #     f"GLOBAL_CONTEXT.max_concurrent_jobs was {GLOBAL_CONTEXT.max_concurrent_jobs}"
                # )
                print(f"Running {len(GLOBAL_CONTEXT.running)} jobs")
            # Someone is waiting (maybe us), don't start any more jobs
            GLOBAL_CONTEXT.max_concurrent_jobs = len(GLOBAL_CONTEXT.running) - 1
    MAX_CONCURRENT_JOBS = 100
    if GLOBAL_CONTEXT.max_concurrent_jobs > MAX_CONCURRENT_JOBS:
        GLOBAL_CONTEXT.max_concurrent_jobs = MAX_CONCURRENT_JOBS


seeds = list(range(10))

mujoco_env_names_v3 = [
    "HalfCheetah-v3",
    "Walker2d-v3",
    "Hopper-v3",
    "Swimmer-v3",
    "InvertedPendulum-v2",
    "Reacher-v2",
]

EPOCHS = 500  # Basically, this determines the x-axis plot resolution for Tianshou logs


def mujoco_xppo_tianshou(
    seed, env, group, priority=None, cores=2, add_to_path=None, total_steps=None, **kwargs
):
    if total_steps is None:
        total_steps = 10_000_000
    if priority is None:
        priority = (50, -seed)
    if add_to_path is None:
        add_to_path = [k for k, _ in kwargs.items()][:5]
    kwargs_path = "_".join(
        f"{k.replace('_', '-')}={kwargs.get(k)}" for k in add_to_path
    )
    return cmd(
        "python",
        "src/mujoco_xppo_tianshou.py",
        "--seed",
        seed,
        "--env",
        env,
        "--epoch", EPOCHS,
        "--step-per-epoch", math.ceil(total_steps / EPOCHS),
        *[f"--{k.replace('_', '-')}={v}" for (k, v) in kwargs.items()],
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-group",
        group,
        "--log-dir",
        Out(f"xppo_tianshou/env={env}_seed={seed}_{kwargs_path}_group={group}/"),
        warmup_time=3,
        ram_gb=6,
        priority=priority,
        cores=cores,
    )

if HOST == BRAIN_HOSTNAME:
    # for seed in seeds:
    #     for env in mujoco_env_names_v3:
    #             total_steps = 10_000_000
    #             group = "xppo-tianshou-mujoco-core-profile"
    #             cores = seed + 2
    #             cmd(
    #                 "python",
    #                 "src/mujoco_xppo_tianshou.py",
    #                 "--seed",
    #                 seed,
    #                 "--env",
    #                 env,
    #                 "--epoch", EPOCHS,
    #                 "--step-per-epoch", math.ceil(total_steps / EPOCHS),
    #                 "--wandb-entity", WANDB_ENTITY,
    #                 "--wandb-group",
    #                 "xppo-tianshou-mujoco-core-profile",
    #                 "--log-dir",
    #                 Out(f"xppo_tianshou/env={env}_seed={seed}_cores={cores}_group={group}/"),
    #                 warmup_time=3,
    #                 ram_gb=6,
    #                 priority=(60, -seed),
    #                 cores=cores,
    #             )
    for seed in seeds:
        for env_i, env in enumerate(mujoco_env_names_v3):
            for eps_kl_args in [{"eps_kl": 0.2}, {}, {"eps_kl": 1.0}]:
                for target_coeff in [1, 2, 3, 5, 10]:
                    mujoco_xppo_tianshou(
                        seed=seed,
                        env=env,
                        group="xppo-tianshou-mujoco",
                        target_coeff=target_coeff,
                        **eps_kl_args,
                        priority=(50, -env_i, -seed, -target_coeff))
                    mujoco_xppo_tianshou(
                        seed=seed,
                        env=env,
                        group="xppo-tianshou-mujoco",
                        fixup_every_repeat=0,
                        target_coeff=target_coeff,
                        **eps_kl_args,
                        priority=(51, -env_i, -seed))
                mujoco_xppo_tianshou(
                    seed=seed,
                    env=env,
                    group="xppo-tianshou-mujoco",
                    fixup_loop=0,
                    **eps_kl_args,
                    priority=(50, -env_i, -seed))
                mujoco_xppo_tianshou(
                    seed=seed,
                    env=env,
                    group="xppo-tianshou-mujoco",
                    kl_target_stat="mean",
                    **eps_kl_args,
                    priority=(50, -env_i, -seed))
