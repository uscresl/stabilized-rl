from doexp import cmd, In, Out, GLOBAL_CONTEXT
import psutil
from socket import gethostname

HOST = gethostname()

if HOST == 'brain.usc.edu':
    GLOBAL_CONTEXT.max_concurrent_jobs = 8

mujoco_envs = ['InvertedDoublePendulum-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
seeds = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
for seed in seeds:
    for env in mujoco_envs:
        for center_adv in [True, False]:
            cmd(
                "python",
                "src/ppo_mujoco.py",
                "--seed", seed,
                "--env", env,
                f"--center-adv={center_adv}",
                "--log-dir", Out(f"ppo/env={env}_seed={seed}_center-adv={center_adv}/"),
                warmup_time=3,
                ram_gb=20,
            )
        cmd(
            "python",
            "src/ppo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--center-adv=True",
            "--normalize-env",
            "--log-dir", Out(f"ppo/env={env}_seed={seed}_normalized/"),
            warmup_time=3,
            ram_gb=20,
        )
        cmd(
            "python",
            "src/ppo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--center-adv=True",
            "--normalize-env",
            "--use-vec-worker",
            "--log-dir", Out(f"ppo/env={env}_seed={seed}_normalized_vecw/"),
            warmup_time=3,
            ram_gb=20,
        )
        cmd(
            "python",
            "src/klpo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--log-dir", Out(f"klpo/env={env}_seed={seed}/"),
            warmup_time=3,
            ram_gb=20,
        )
        cmd(
            "python",
            "src/klpo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--normalize-env",
            "--log-dir", Out(f"klpo/env={env}_seed={seed}_normalized/"),
            warmup_time=3,
            ram_gb=20,
        )
        cmd(
            "python",
            "src/klpo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--lr-loss-coeff=0.5", 
            "--log-dir", Out(f"klpo/env={env}_seed={seed}_lr-loss-coeff=0.5/"),
            warmup_time=3,
            ram_gb=20,
        )
        cmd(
            "python",
            "src/klpo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--lr-loss-coeff=0.05", 
            "--log-dir", Out(f"klpo/env={env}_seed={seed}_lr-loss-coeff=0.05/"),
            warmup_time=3,
            ram_gb=20,
        )
        cmd(
            "python",
            "src/klpo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--lr-loss-coeff=0.3", 
            "--log-dir", Out(f"klpo/env={env}_seed={seed}_lr-loss-coeff=0.3/"),
            warmup_time=3,
            ram_gb=20,
        )
        cmd(
            "python",
            "src/klpo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--lr-loss-coeff=0.7", 
            "--log-dir", Out(f"klpo/env={env}_seed={seed}_lr-loss-coeff=0.7/"),
            warmup_time=3,
            ram_gb=20,
        )
