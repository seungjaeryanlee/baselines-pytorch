from gym.envs.registration import register
import gym
import numpy as np

# import gym_minigrid


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

def get_lunarlander(seed=None):
    env = gym.make('LunarLander-v2')

    if seed:
        env.seed(seed)

    return env