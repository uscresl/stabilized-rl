from tianshou.env import ShmemVectorEnv, VectorEnvNormObs

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gym.wrappers import TimeLimit

def gen_env(env_name: str):
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-v2-goal-observable"](seed=0)
    env.seeded_rand_vec = True
    env = TimeLimit(env, max_episode_steps=500)
    return env

def make_metaworld_env(task, seed, training_num, test_num, obs_norm):
    env = gen_env(task)
    train_envs = ShmemVectorEnv(
        [lambda: gen_env(task) for _ in range(training_num)]
    )
    test_envs = ShmemVectorEnv([lambda: gen_env(task) for _ in range(test_num)])
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs
