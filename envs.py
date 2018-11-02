from gym.envs.registration import register
import gym
import numpy as np

import gym_minigrid


def register_env():
    register(
        id='FrozenLakeDeterministic8x8-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '8x8', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )

def get_frozenlake(seed=None):
    register_env()
    env = gym.make('FrozenLakeDeterministic8x8-v0')

    if seed:
        env.seed(seed)

    return env

class MinigridWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
    
    def reset(self):
        obs = self.env.reset()
        obs = obs['image'].transpose(2, 0, 1)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = obs['image'].transpose(2, 0, 1)
        return obs, rew, done, info


# TODO Support this environment
def get_minigrid(seed=None):
    env = gym.make('MiniGrid-Empty-6x6-v0')

    if seed:
        env.seed(seed)

    return MinigridWrapper(env)
