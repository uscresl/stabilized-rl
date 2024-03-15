# Fixup Policy Optimization (FixPO)

This repo contains the code for ["Guaranteed Trust Region Optimization via Two-Phase KL Penalization"](https://arxiv.org/abs/2312.05405).

It implements an efficient trust region optimization algorithm, FixPO.

## Setup Instructions

Install Python >=3.8

Install Python's poetry dependency manager:

```
curl -sSL https://install.python-poetry.org | python3 -
```

Install Mujoco 2.1:

```
mkdir -p $HOME/.mujoco && \
curl -O https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
tar xf mujoco210-linux-x86_64.tar.gz --directory $HOME/.mujoco

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
source ~/.bashrc
```

Install dependencies:

```
pip install --upgrade --user pip
poetry install
```

Alternatively, use `install.sh`.

There are subdirectories containing `tianshou v0.5.0`, `Meta-World v2.0.0`, and
`trust-region-layers` in this repo that are patched to be compatible with each other.
See `patches` if you would like to apply these changes yourself.

Most experiments are listed in `exps_tianshou.py` and use code in `src`.

To run those experiments:

```
poetry run doexp exps_tianshou.py
```

## File Overview

`exps_tianshou.py`: Defines all of the experiments to run that use tianshou.

`exps_trust_region_layers.py`: Defines all of the experiments to run that use trust_region_layers.

`doexp`: Runs all experiments defined in `exps.py`


`src/fixpo_tianshou.py`: Main algorithm code.

`src/mujoco_fixpo_tianshou.py`: Launcher for FixPO with Gym environments.

`src/metaworld_fixpo_tianshou.py`: Launcher for FixPO with Meta-World environments.

`src/mujoco_ppo_tianshou.py`: Launcher for PPO with Gym environments.

`src/mujoco_trpo_tianshou.py`: Launcher for TRPO with Gym environments.

`src/metaworld_ppo_tianshou.py`: Launcher for PPO with Meta-World environments.

`src/metaworld_utils.py`: Helper function for setting up Meta-World environments.

`src/metaworld_env_tianshou.py`: Helper function for setting up Meta-World environments.

`patches/tianshou.patch`: Patch to Tianshou v0.5.0 that allows running Meta-World experiments.

`patches/metaworld.patch`: Patch to Meta-World v2.0.0 that allows running Meta-World experiments.

`patches/trust_region_layers.patch`: Patch to trust_region_layers that allows running Meta-World experiments.

`patches/mujoco_kl_config.json`: Base config to use the KL projection proposed in  trust_region_layers.


## Citation:

If you use this code in your research, please cite:
```
@misc{zentner2023guaranteed,
  title={Guaranteed Trust Region Optimization via Two-Phase KL Penalization},
  author={K. R. Zentner and Ujjwal Puri and Zhehui Huang and Gaurav S. Sukhatme},
  year={2023},
  eprint={2312.05405},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
