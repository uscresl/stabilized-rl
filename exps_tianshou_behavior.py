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
    # MIN_CONCURRENT_JOBS = 100
    MIN_CONCURRENT_JOBS = 50
    # MIN_CONCURRENT_JOBS = 30
    GLOBAL_CONTEXT.max_core_alloc = 600
    squeue_res = run(["squeue", "--all"], check=False, capture_output=True)
    if squeue_res.returncode == 0:
        out = squeue_res.stdout.decode()
        waiting_job_count = 0
        for line in out.split("\n"):
            if "(Priority)" in line or "(Resources)" in line:
                # if "slurm-lon" not in line:
                #     waiting_job_count += 1
                # if " kr " in line:
                waiting_job_count += 1
        if waiting_job_count <= 1:
            # Presumably free resources on the cluster
            GLOBAL_CONTEXT.max_concurrent_jobs += 1
        else:
            # print("waiting_job_count =", waiting_job_count)
            if GLOBAL_CONTEXT.max_concurrent_jobs > len(GLOBAL_CONTEXT.running):
                print(f"Running {len(GLOBAL_CONTEXT.running)} jobs")
            # Someone is waiting (maybe us), don't start any more jobs
            # GLOBAL_CONTEXT.max_concurrent_jobs = len(GLOBAL_CONTEXT.running) - 1
            GLOBAL_CONTEXT.max_concurrent_jobs = MIN_CONCURRENT_JOBS
    if GLOBAL_CONTEXT.max_concurrent_jobs < MIN_CONCURRENT_JOBS:
        GLOBAL_CONTEXT.max_concurrent_jobs = MIN_CONCURRENT_JOBS
    # MAX_CONCURRENT_JOBS = 300
    MAX_CONCURRENT_JOBS = 170
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

for seed in seeds:
    if seed < 3:
        base_priority = 60
    else:
        base_priority = 40
    for env_i, env in enumerate(MT50_ENV_NAMES):
        group = "fixpo-tianshou-metaworld-behavior"
        cmd(
            "python",
            "src/metaworld_fixpo_debug_tianshou.py",
            "--seed",
            seed,
            "--env",
            env,
            "--epoch", 100,
            "--wandb-entity", WANDB_ENTITY,
            "--wandb-group", group,
            "--gen-behavior", 1,
            "--log-dir",
            Out(f"fixpo_tianshou/env={env}_seed={seed}_group={group}/"),
            warmup_time=1,
            ram_gb=6,
            priority=(base_priority, -seed, -env_i),
            cores=2,
        )
    for env_i, env in enumerate(MT10_ENV_NAMES):
        if env == 'pick-place':
            env_priority = 10
        else:
            env_priority = 0
        for base_env in MT10_ENV_NAMES:
            group = "fixpo-tianshou-metaworld-transfer-behavior"
            cmd(
                "python",
                "src/metaworld_fixpo_debug_tianshou.py",
                "--seed",
                seed,
                "--env",
                env,
                "--epoch", 200,
                "--base-task-path", In(f"fixpo_tianshou/env={base_env}_seed={seed}_group=fixpo-tianshou-metaworld-behavior/policy.pth"),
                "--wandb-entity", WANDB_ENTITY,
                "--wandb-group", group,
                "--gen-behavior", 1,
                "--log-dir",
                Out(f"fixpo_tianshou/env={env}_base-env={base_env}_seed={seed}_group={group}/"),
                warmup_time=1,
                ram_gb=6,
                priority=(base_priority - 5, env_priority, -seed, -env_i),
                cores=2,
            )
            # back_group = "fixpo-tianshou-metaworld-transfer-back-behavior"
            # cmd(
            #     "python",
            #     "src/metaworld_fixpo_debug_tianshou.py",
            #     "--seed",
            #     seed,
            #     "--env",
            #     "pick-place",
            #     "--epoch", 100,
            #     "--base-task-path", In(f"fixpo_tianshou/env={env}_seed={seed}_group={group}/policy.pth"),
            #     "--wandb-entity", WANDB_ENTITY,
            #     "--wandb-group", group,
            #     "--gen-behavior", 1,
            #     "--log-dir",
            #     Out(
            #         f"fixpo_tianshou/env=pick-place_base_env={env}_seed={seed}_group={back_group}/"
            #     ),
            #     warmup_time=1,
            #     ram_gb=6,
            #     priority=(base_priority + 10, -seed, -env_i),
            #     cores=2,
            # )
