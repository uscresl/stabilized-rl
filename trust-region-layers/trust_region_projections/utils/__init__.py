#   Copyright (c) 2021 Robert Bosch GmbH
#   Author: Fabian Otto
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.


def make_env(env_name: str):
    if env_name.startswith('metaworld-'):
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        from gym.wrappers import TimeLimit

        env_name = env_name[len('metaworld-'):]
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-v2-goal-observable"](seed=0)
        env.seeded_rand_vec = True
        env.add_reset_info = False
        env = TimeLimit(env, max_episode_steps=500)
        return env
    else:
        import gym
        return gym.make(env_name)
