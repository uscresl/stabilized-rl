from doexp import cmd, In, Out, GLOBAL_CONTEXT

atari_env_names = [
    "PongNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    # "BeamRiderNoFrameskip-v4",
    # "SpaceInvadersNoFrameskip-v4",
]
seeds = list(range(10))

for seed in seeds:
    for env_i, env in enumerate(atari_env_names):
        group = "xppo-atari"
        cmd("python", "src/atari_xppo_tianshou.py", "--log-dir",
            Out(f"xppo_tianshou/env={env}_seed={seed}_group={group}/"),
            "--wandb-group", group,
            gpu_ram_gb=2.5, cores=8, ram_gb=8,
            priority=(50, -env_i, -seed))
