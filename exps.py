from doexp import cmd, In, Out, GLOBAL_CONTEXT
import psutil
from socket import gethostname

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

for seed in seeds:
    target_kl = 0.2
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
            ram_gb=20,
            priority=-10,
        )
