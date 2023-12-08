from doexp import cmd, In, Out, GLOBAL_CONTEXT
from socket import gethostname
from subprocess import run
import math
import os


HOST = gethostname()

WANDB_ENTITY = ""
if "WANDB_ENTITY" in os.environ:
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
else:
    print("Please set WANDB_ENTITY")
os.environ["WANDB__SERVICE_WAIT"] = "300"
SLURM_HOSTNAME = "brain.usc.edu"

if HOST == SLURM_HOSTNAME:
    MIN_CONCURRENT_JOBS = 30
    GLOBAL_CONTEXT.max_core_alloc = 600
    squeue_res = run(["squeue", "--all"], check=False, capture_output=True)
    if squeue_res.returncode == 0:
        out = squeue_res.stdout.decode()
        waiting_job_count = 0
        for line in out.split("\n"):
            if "(Priority)" in line or "(Resources)" in line:
                if "slurm-lon" not in line:
                    waiting_job_count += 1
        if waiting_job_count <= 1:
            # Presumably free resources on the cluster
            GLOBAL_CONTEXT.max_concurrent_jobs += 1
        else:
            # print("waiting_job_count =", waiting_job_count)
            if GLOBAL_CONTEXT.max_concurrent_jobs > len(GLOBAL_CONTEXT.running):
                print(f"Running {len(GLOBAL_CONTEXT.running)} jobs")
            # Someone is waiting (maybe us), don't start any more jobs
            GLOBAL_CONTEXT.max_concurrent_jobs = len(GLOBAL_CONTEXT.running) - 1
    if GLOBAL_CONTEXT.max_concurrent_jobs < MIN_CONCURRENT_JOBS:
        GLOBAL_CONTEXT.max_concurrent_jobs = MIN_CONCURRENT_JOBS
    MAX_CONCURRENT_JOBS = 300
    # MAX_CONCURRENT_JOBS = 2
    if GLOBAL_CONTEXT.max_concurrent_jobs > MAX_CONCURRENT_JOBS:
        GLOBAL_CONTEXT.max_concurrent_jobs = MAX_CONCURRENT_JOBS

# print(f"Running {len(GLOBAL_CONTEXT.running)} jobs")

seeds = list(range(10))

mujoco_env_names_v3 = [
    "HalfCheetah-v3",
    "Walker2d-v3",
    "Hopper-v3",
    "Swimmer-v3",
    "InvertedPendulum-v2",
    "InvertedDoublePendulum-v2",
    "Reacher-v2",
]

EPOCHS = 500  # Basically, this determines the x-axis plot resolution for Tianshou logs


MT50_ENV_NAMES = [
    "assembly",
    "basketball",
    "bin-picking",
    "box-close",
    "button-press-topdown",
    "button-press-topdown-wall",
    "button-press",
    "button-press-wall",
    "coffee-button",
    "coffee-pull",
    "coffee-push",
    "dial-turn",
    "disassemble",
    "door-close",
    "door-lock",
    "door-open",
    "door-unlock",
    "hand-insert",
    "drawer-close",
    "drawer-open",
    "faucet-open",
    "faucet-close",
    "hammer",
    "handle-press-side",
    "handle-press",
    "handle-pull-side",
    "handle-pull",
    "lever-pull",
    "peg-insert-side",
    "pick-place-wall",
    "pick-out-of-hole",
    "reach",
    "push-back",
    "push",
    "pick-place",
    "plate-slide",
    "plate-slide-side",
    "plate-slide-back",
    "plate-slide-back-side",
    "peg-unplug-side",
    "soccer",
    "stick-push",
    "stick-pull",
    "push-wall",
    "reach-wall",
    "shelf-place",
    "sweep-into",
    "sweep",
    "window-open",
    "window-close",
]

MT10_ENV_NAMES = [
    'reach',
    'push',
    'pick-place',
    'door-open',
    'drawer-open',
    'drawer-close',
    'button-press-topdown',
    'peg-insert-side',
    'window-open',
    'window-close',
]


def mujoco_fixpo_tianshou(
    seed,
    env,
    group,
    priority=None,
    cores=2,
    add_to_path=None,
    total_steps=None,
    **kwargs,
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
        "src/mujoco_fixpo_tianshou.py",
        "--seed",
        seed,
        "--env",
        env,
        "--epoch",
        EPOCHS,
        "--step-per-epoch",
        math.ceil(total_steps / EPOCHS),
        *[f"--{k.replace('_', '-')}={v}" for (k, v) in kwargs.items()],
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-group",
        group,
        "--log-dir",
        Out(f"fixpo_tianshou/env={env}_seed={seed}_{kwargs_path}_group={group}/"),
        warmup_time=3,
        ram_gb=6,
        priority=priority,
        cores=cores,
    )


def mujoco_ppo_tianshou(
    seed,
    env,
    group,
    priority=None,
    cores=2,
    add_to_path=None,
    total_steps=None,
    **kwargs,
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
        "src/mujoco_ppo_tianshou.py",
        "--seed",
        seed,
        "--env",
        env,
        "--epoch",
        EPOCHS,
        "--step-per-epoch",
        math.ceil(total_steps / EPOCHS),
        *[f"--{k.replace('_', '-')}={v}" for (k, v) in kwargs.items()],
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-group",
        group,
        "--log-dir",
        Out(f"ppo_tianshou/env={env}_seed={seed}_{kwargs_path}_group={group}/"),
        warmup_time=3,
        ram_gb=6,
        priority=priority,
        cores=cores,
    )


def mujoco_trpo_tianshou(
    seed,
    env,
    group,
    priority=None,
    cores=2,
    add_to_path=None,
    total_steps=None,
    **kwargs,
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
        "src/mujoco_trpo_tianshou.py",
        "--seed",
        seed,
        "--env",
        env,
        "--epoch",
        EPOCHS,
        "--step-per-epoch",
        math.ceil(total_steps / EPOCHS),
        *[f"--{k.replace('_', '-')}={v}" for (k, v) in kwargs.items()],
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-group",
        group,
        "--log-dir",
        Out(f"trpo_tianshou/env={env}_seed={seed}_{kwargs_path}_group={group}/"),
        warmup_time=3,
        ram_gb=6,
        priority=priority,
        cores=cores,
    )


def metaworld_fixpo_tianshou(
    seed,
    env,
    group,
    priority=None,
    cores=2,
    add_to_path=None,
    total_steps=None,
    **kwargs,
):
    if total_steps is None:
        total_steps = 20_000_000
    if priority is None:
        priority = (50, -seed)
    if add_to_path is None:
        add_to_path = [k for k, _ in kwargs.items()][:5]
    kwargs_path = "_".join(
        f"{k.replace('_', '-')}={kwargs.get(k)}" for k in add_to_path
    )
    return cmd(
        "python",
        "src/metaworld_fixpo_tianshou.py",
        "--seed",
        seed,
        "--env",
        env,
        "--epoch",
        EPOCHS,
        "--step-per-epoch",
        math.ceil(total_steps / EPOCHS),
        *[f"--{k.replace('_', '-')}={v}" for (k, v) in kwargs.items()],
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-group",
        group,
        "--log-dir",
        Out(f"fixpo_tianshou/env={env}_seed={seed}_{kwargs_path}_group={group}/"),
        warmup_time=3,
        ram_gb=6,
        priority=priority,
        cores=cores,
    )


def metaworld_ppo_tianshou(
    seed,
    env,
    group,
    priority=None,
    cores=2,
    add_to_path=None,
    total_steps=None,
    **kwargs,
):
    if total_steps is None:
        total_steps = 20_000_000
    if priority is None:
        priority = (50, -seed)
    if add_to_path is None:
        add_to_path = [k for k, _ in kwargs.items()][:5]
    kwargs_path = "_".join(
        f"{k.replace('_', '-')}={kwargs.get(k)}" for k in add_to_path
    )
    return cmd(
        "python",
        "src/metaworld_ppo_tianshou.py",
        "--seed",
        seed,
        "--env",
        env,
        "--epoch",
        EPOCHS,
        "--step-per-epoch",
        math.ceil(total_steps / EPOCHS),
        *[f"--{k.replace('_', '-')}={v}" for (k, v) in kwargs.items()],
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-group",
        group,
        "--log-dir",
        Out(f"ppo_tianshou/env={env}_seed={seed}_{kwargs_path}_group={group}/"),
        warmup_time=3,
        ram_gb=6,
        priority=priority,
        cores=cores,
    )


if HOST == SLURM_HOSTNAME:
    # for seed in seeds:
    #     for env in mujoco_env_names_v3:
    #             total_steps = 10_000_000
    #             group = "fixpo-tianshou-mujoco-core-profile"
    #             cores = seed + 2
    #             cmd(
    #                 "python",
    #                 "src/mujoco_fixpo_tianshou.py",
    #                 "--seed",
    #                 seed,
    #                 "--env",
    #                 env,
    #                 "--epoch", EPOCHS,
    #                 "--step-per-epoch", math.ceil(total_steps / EPOCHS),
    #                 "--wandb-entity", WANDB_ENTITY,
    #                 "--wandb-group",
    #                 "fixpo-tianshou-mujoco-core-profile",
    #                 "--log-dir",
    #                 Out(f"fixpo_tianshou/env={env}_seed={seed}_cores={cores}_group={group}/"),
    #                 warmup_time=3,
    #                 ram_gb=6,
    #                 priority=(60, -seed),
    #                 cores=cores,
    #             )
    for seed in seeds[:3]:
        if seed < 3:
            base_priority = 60
        else:
            base_priority = 40
            continue
        for env_i, env in enumerate(MT50_ENV_NAMES):
            metaworld_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-metaworld",
                step_per_collect=10_000,
                priority=(base_priority, -seed, -env_i),
            )
            metaworld_ppo_tianshou(
                seed=seed,
                env=env,
                group="ppo-tianshou-metaworld",
                max_grad_norm=0.1,
                step_per_collect=10_000,
                priority=(base_priority, -seed, -env_i),
            )
        for env_i, env in enumerate(MT10_ENV_NAMES):
            group = "fixpo-tianshou-metaworld-transfer"
            cmd(
                "python",
                "src/metaworld_fixpo_tianshou.py",
                "--seed",
                seed,
                "--env",
                env,
                "--epoch", 100,
                "--base-task-path", In(f"fixpo_tianshou/env=pick-place_seed={seed}_step-per-collect=10000_group=fixpo-tianshou-metaworld/policy.pth"),
                "--wandb-entity", WANDB_ENTITY,
                "--wandb-group", group,
                "--log-dir",
                Out(f"fixpo_tianshou/env={env}_seed={seed}_group={group}/"),
                warmup_time=3,
                ram_gb=6,
                priority=(base_priority - 5, -seed, -env_i),
                cores=2,
            )
            back_group = "fixpo-tianshou-metaworld-transfer-back"
            cmd(
                "python",
                "src/metaworld_fixpo_tianshou.py",
                "--seed",
                seed,
                "--env",
                "pick-place",
                "--epoch", 100,
                "--base-task-path", In(f"fixpo_tianshou/env={env}_seed={seed}_group={group}/policy.pth"),
                "--wandb-entity", WANDB_ENTITY,
                "--wandb-group", group,
                "--log-dir",
                Out(
                    f"fixpo_tianshou/env=pick-place_base_env={env}_seed={seed}_group={back_group}/"
                ),
                warmup_time=3,
                ram_gb=6,
                priority=(base_priority + 10, -seed, -env_i),
                cores=2,
            )

            # metaworld_fixpo_tianshou(
            #     seed=seed,
            #     env=env,
            #     group="fixpo-tianshou-metaworld",
            #     step_per_collect=50_000,
            #     priority=(61, -seed, -env_i),
            # )
            # metaworld_fixpo_tianshou(
            #     seed=seed,
            #     env=env,
            #     group="fixpo-tianshou-metaworld",
            #     fixup_every_repeat=0,
            #     priority=(60, -seed, -env_i),
            # )
            # metaworld_ppo_tianshou(
            #     seed=seed,
            #     env=env,
            #     group="ppo-tianshou-metaworld",
            #     priority=(61, -seed, -env_i),
            # )
        for env_i, env in enumerate(MT10_ENV_NAMES):
            group = "ppo-tianshou-metaworld-transfer"
            hidden_sizes = [128,128]
            if seed in [0, 1, 2]:
                hidden_sizes = [64,64]
            # print("data/" + f"ppo_tianshou/env=pick-place_seed={seed}_step-per-collect=10000_group=ppo-tianshou-metaworld/policy.pth")
            cmd(
                "python",
                "src/metaworld_ppo_tianshou.py",
                "--seed",
                seed,
                "--env",
                env,
                "--epoch", 100,
                "--hidden-sizes", *hidden_sizes,
                "--base-task-path", In(f"ppo_tianshou/env=pick-place_seed={seed}_step-per-collect=10000_group=ppo-tianshou-metaworld/policy.pth"),
                "--wandb-entity", WANDB_ENTITY,
                "--wandb-group", group,
                "--log-dir",
                Out(f"ppo_tianshou/env={env}_seed={seed}_group={group}/"),
                warmup_time=3,
                ram_gb=6,
                priority=(base_priority + 5, -seed, -env_i),
                cores=2,
            )
            back_group = "ppo-tianshou-metaworld-transfer-back"
            cmd(
                "python",
                "src/metaworld_ppo_tianshou.py",
                "--seed",
                seed,
                "--env",
                "pick-place",
                "--epoch", 100,
                "--hidden-sizes", *hidden_sizes,
                "--base-task-path", In(f"ppo_tianshou/env={env}_seed={seed}_group={group}/policy.pth"),
                "--wandb-entity", WANDB_ENTITY,
                "--wandb-group", group,
                "--log-dir",
                Out(f"ppo_tianshou/env=pick-place_base_env={env}_seed={seed}_group={back_group}/"),
                warmup_time=3,
                ram_gb=6,
                priority=(base_priority + 10, -seed, -env_i),
                cores=2,
            )

    for seed in seeds:
        for env_i, env in enumerate(mujoco_env_names_v3):
            mujoco_ppo_tianshou(
                seed=seed,
                env=env,
                group="ppo-tianshou-mujoco",
                priority=(100, -env_i, -seed))

            mujoco_trpo_tianshou(
                seed=seed,
                env=env,
                group="trpo-tianshou-mujoco",
                priority=(200, -env_i, -seed))

            target_coeff = 3
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-mujoco",
                target_coeff=target_coeff,
                priority=(100, -env_i, -seed))
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-mujocob-50k",
                target_coeff=target_coeff,
                step_per_collect=50_000,
                priority=(60, -env_i, -seed))
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-mujoco",
                fixup_loop=0,
                target_coeff=target_coeff,
                priority=(60, -env_i, -seed),
            )
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-mujoco",
                fixup_every_repeat=0,
                target_coeff=target_coeff,
                priority=(60, -env_i, -seed),
            )
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-mujoco",
                kl_target_stat="mean",
                target_coeff=target_coeff,
                priority=(60, -env_i, -seed),
            )
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-mujoco",
                kl_target_stat="max",
                target_coeff=1,
                priority=(60, -env_i, -seed),
            )
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-mujoco",
                init_beta=10,
                beta_lr=0,
                priority=(60, -env_i, -seed),
            )
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-tianshou-mujoco",
                kl_target_stat="mean",
                fixup_loop=0,
                target_coeff=target_coeff,
                priority=(30, -env_i, -seed),
            )

            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                group="fixpo-mujoco",
                init_beta=10,
                beta_lr=0,
                fixup_loop=0,
                priority=(160, -env_i, -seed),
            )


    for seed in seeds[:3]:
        for env_i, env in enumerate(MT50_ENV_NAMES):
            group = "fixpo-tianshou-beta-distribution-metaworld"
            metaworld_fixpo_tianshou(
                seed=seed,
                env=env,
                group=group,
                dist="beta",
                step_per_collect=10_000,
                priority=(150, -env_i, -seed),
            )
    for seed in seeds:
        for env_i, env in enumerate(mujoco_env_names_v3):
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                dist="beta",
                group="fixpo-tianshou-beta-distribution-mujoco",
                priority=(160, -env_i, -seed),
            )

            # for eps_kl_args in [{"eps_kl": 0.2}, {}, {"eps_kl": 1.0}]:
            #     for target_coeff in [2, 3, 5]:
            #         mujoco_fixpo_tianshou(
            #             seed=seed,
            #             env=env,
            #             group="fixpo-tianshou-mujoco",
            #             target_coeff=target_coeff,
            #             **eps_kl_args,
            #             priority=(50, -env_i, -seed, -target_coeff))
            #         mujoco_fixpo_tianshou(
            #             seed=seed,
            #             env=env,
            #             group="fixpo-tianshou-mujoco",
            #             fixup_every_repeat=0,
            #             target_coeff=target_coeff,
            #             **eps_kl_args,
            #             priority=(51, -env_i, -seed))
            #     mujoco_fixpo_tianshou(
            #         seed=seed,
            #         env=env,
            #         group="fixpo-tianshou-mujoco",
            #         fixup_loop=0,
            #         **eps_kl_args,
            #         priority=(50, -env_i, -seed))
            #     mujoco_fixpo_tianshou(
            #         seed=seed,
            #         env=env,
            #         group="fixpo-tianshou-mujoco",
            #         kl_target_stat="mean",
            #         **eps_kl_args,
            #         priority=(50, -env_i, -seed))
elif HOST == "tanuki":
    GLOBAL_CONTEXT.max_concurrent_jobs = 4
    ram_gb = 4

    for seed in seeds[:1]:
        for env_i, env in enumerate(MT50_ENV_NAMES[:1]):
            group = "fixpo-tianshou-beta-distribution-metaworld"
            metaworld_fixpo_tianshou(
                seed=seed,
                env=env,
                group=group,
                dist="beta",
                step_per_collect=10_000,
                priority=(3, -env_i, -seed),
            )
        for env_i, env in enumerate(mujoco_env_names_v3[:1]):
            mujoco_fixpo_tianshou(
                seed=seed,
                env=env,
                dist="beta",
                group="fixpo-tianshou-beta-distribution-mujoco",
                priority=(60, -env_i, -seed),
            )
