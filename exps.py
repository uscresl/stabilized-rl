from doexp import cmd, In, Out, GLOBAL_CONTEXT
import psutil

mujoco_envs = ['InvertedDoublePendulum-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3']
seeds = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
for seed in seeds:
    for env in mujoco_envs:
        cmd(
            "python",
            "src/ppo_mujoco.py",
            "--seed", seed,
            "--env", env,
            "--log-dir", Out(f"ppo/env={env}_seed={seed}/"),
        )
