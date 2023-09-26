from doexp import cmd, In, Out, GLOBAL_CONTEXT
from socket import gethostname
import json
import sys
from datetime import datetime

HOST = gethostname()
if HOST == "brain.usc.edu":
    GLOBAL_CONTEXT.max_concurrent_jobs = 1


mujoco_envs = [
    "HalfCheetah-v2",
    "Hopper-v2",
    "Walker2d-v2",
    "Swimmer-v3",
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
    "Reacher-v2",
]

with open('trust-region-layers/configs/pg/mujoco_kl_config.json') as f:
    kl_config = json.load(f)

with open('trust-region-layers/configs/pg/mujoco_papi_config.json') as f:
    papi_conf = json.load(f)

for seed in seeds:
    for env in mujoco_env_names_v3:
        conf_name = f"data_tmp/trust-region-layers_kl_seed={seed}_env={env}.json"
        out_dir = f"trust-region-layers_kl_seed={seed}_env={env}/"
        kl_config["n_envs"] = 1  # Should only affect sampling speed
        kl_config["seed"] = seed
        kl_config["game"] = env
        kl_config["out_dir"] = f"data_tmp/{out_dir}"
        kl_config["exp_name"] = f'seed_{seed}_env_{env}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        with open(conf_name, 'w') as f:
            json.dump(kl_config, f, indent=2)
        cmd("python", "trust-region-layers/main.py", conf_name, extra_outputs=[Out(out_dir)],
            cores=3, ram_gb=8)

        conf_name = f"data_tmp/trust-region-layers_papi_seed={seed}_env={env}.json"
        out_dir = f"trust-region-layers_papi_seed={seed}_env={env}/"
        papi_conf["n_envs"] = 1  # Should only affect sampling speed
        papi_conf["seed"] = seed
        papi_conf["game"] = env
        papi_conf["out_dir"] = f"data_tmp/{out_dir}"
        papi_conf["exp_name"] = f'seed_{seed}_env_{env}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        with open(conf_name, 'w') as f:
            json.dump(papi_conf, f, indent=2)
        cmd("python", "trust-region-layers/main.py", conf_name, extra_outputs=[Out(out_dir)],
            cores=3, ram_gb=8)
