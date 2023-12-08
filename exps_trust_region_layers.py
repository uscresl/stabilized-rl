from doexp import cmd, In, Out, GLOBAL_CONTEXT
from socket import gethostname
from subprocess import run
import json
from datetime import datetime
import os

HOST = gethostname()

WANDB_ENTITY = ""
if "WANDB_ENTITY" in os.environ:
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
else:
    print("Please set WANDB_ENTITY")
os.environ["WANDB__SERVICE_WAIT"] = "300"


mujoco_envs = [
    "HalfCheetah-v2",
    "Hopper-v2",
    "Walker2d-v2",
    "Swimmer-v3",
    "InvertedPendulum-v2",
    "InvertedDoublePendulum-v2",
    "Reacher-v2",
]

mujoco_envs_remaining = [
    "Swimmer-v2",
    "InvertedPendulum-v2",
    "InvertedDoublePendulum-v2",
    "Reacher-v2",
]

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


with open('trust-region-layers/configs/pg/mujoco_kl_config.json') as f:
    kl_config = json.load(f)

with open('trust-region-layers/configs/pg/mujoco_papi_config.json') as f:
    papi_conf = json.load(f)

for seed in seeds:
    for env in mujoco_envs_remaining:
        conf_name = f"data_tmp/trust-region-layers_kl_seed={seed}_env={env}.json"
        out_dir = f"trust-region-layers_kl_seed={seed}_env={env}/"
        kl_config["n_envs"] = 1  # Should only affect sampling speed
        kl_config["seed"] = seed
        kl_config["game"] = env
        kl_config["out_dir"] = f"data_tmp/{out_dir}"
        kl_config["exp_name"] = f'seed_{seed}_env_{env}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        with open(conf_name, 'w') as f:
            json.dump(kl_config, f, indent=2)
        cmd("python", "trust-region-layers/main.py", conf_name, "--wandb-group=trust-region-layers-papi", extra_outputs=[Out(out_dir)],
            cores=1, ram_gb=6, priority=(51, -seed))

        conf_name = f"data_tmp/trust-region-layers_papi_seed={seed}_env={env}.json"
        out_dir = f"trust-region-layers_papi_seed={seed}_env={env}/"
        papi_conf["n_envs"] = 1  # Should only affect sampling speed
        papi_conf["seed"] = seed
        papi_conf["game"] = env
        papi_conf["out_dir"] = f"data_tmp/{out_dir}"
        papi_conf["exp_name"] = f'seed_{seed}_env_{env}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        with open(conf_name, 'w') as f:
            json.dump(papi_conf, f, indent=2)
        cmd("python", "trust-region-layers/main.py", conf_name, "--wandb-group=trust-region-layers-papi", extra_outputs=[Out(out_dir)],
            cores=1, ram_gb=6, priority=(10, -seed))

for seed in [1, 2, 3]:
    for env_i, env in enumerate(MT50_ENV_NAMES):
        env = "metaworld-" + env
        mt_kl_conf = kl_config.copy()
        conf_name = f"data_tmp/trust-region-layers_kl_seed={seed}_env={env}_logged.json"
        out_dir = f"trust-region-layers_kl_seed={seed}_env={env}/"
        mt_kl_conf["n_envs"] = 10  # Should only affect sampling speed
        mt_kl_conf["n_test_envs"] = 10  # Should only affect sampling speed
        mt_kl_conf["seed"] = seed
        mt_kl_conf["game"] = env
        mt_kl_conf["out_dir"] = f"data_tmp/{out_dir}"
        mt_kl_conf["exp_name"] = f'seed_{seed}_env_{env}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'

        # Meta-World Specific
        mt_kl_conf["rollout_steps"] = 50_000
        mt_kl_conf["max_entropy_coeff"] = 0.01
        mt_kl_conf["max_episode_length"] = 500
        mt_kl_conf["epochs"] = 10
        mt_kl_conf["train_steps"] = 200
        mt_kl_conf["hidden_sizes_policy"] = [128, 128]
        mt_kl_conf["hidden_sizes_vf"] = [128, 128]

        with open(conf_name, 'w') as f:
            json.dump(mt_kl_conf, f, indent=2)
        cmd("python", "trust-region-layers/main.py", conf_name, "--wandb-group=trust-region-layers-kl-metaworld-logged", extra_outputs=[Out(out_dir)],
            cores=2, ram_gb=8, priority=(20, -seed, -env_i))
