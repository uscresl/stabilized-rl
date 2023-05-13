import clize

from garage import wrap_experiment
from stable_baselines3.common.logger import configure
from ppo_stable_baselines_algo import PPO


@wrap_experiment(name_parameters="all", use_existing_dir=True)
def ppo_stbl(
    ctxt,
    env,
    seed,
    note,
    total_steps,
):

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        max_path_length=1000,
    )

    new_logger = configure(ctxt.snapshot_dir, ["stdout", "log", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_steps)
    del note

def run_ppo_stbl(*, log_dir, env: str, seed: int, note: str, total_steps: int):
    ppo_stbl(dict(log_dir=log_dir), env=env, seed=seed, note=note, total_steps=total_steps)

if __name__ == "__main__":
    clize.run(run_ppo_stbl)
