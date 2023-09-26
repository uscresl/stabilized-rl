from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gym.wrappers import TimeLimit

def make_env(env_name: str):
    if env_name.startswith('metaworld-'):
        env_name = env_name[len('metaworld-'):]
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-v2-goal-observable"](seed=0)
    env.seeded_rand_vec = True
    env = TimeLimit(env, max_episode_steps=500)
    return env
