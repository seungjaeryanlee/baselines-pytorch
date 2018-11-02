#!/usr/bin/env python3
"""
Run a specified agent on MiniGrid environment.

TODO This environment is not supported yet.
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import random

from agents import *
from common import get_running_mean
from envs import get_minigrid
from networks import MinigridNetwork
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.optim as optim


# Hyperparameters
NB_TRAINING_EPISODES = 100
LEARNING_RATE = 0.1
EPSILON = 0.1

# For reproducibility
SEED = 0xc0ffee
random.seed(SEED)
np.random.seed(SEED)

# Setup Environment
env = get_minigrid(seed=SEED)

# Setup Agents
q_network = MinigridNetwork()
optimizer = optim.RMSprop(q_network.parameters(), lr=LEARNING_RATE)
online_agent = ApproximateQLearningAgent(q_network, optimizer, 
                                         env.observation_space,
                                         env.action_space, epsilon=EPSILON)

agents = [
    (online_agent, 'Online'),
]


for agent, label in agents:

    # Train agent
    init_state_values = []
    for i in range(NB_TRAINING_EPISODES):
        obs = env.reset()
        init_state_value = agent.get_state_value(obs)
        init_state_values.append(init_state_value)
        print('Initial state value for episode {:5d}: {}'.format(i + 1, init_state_value))
        
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, rew, done, info = env.step(action)
            agent.learn(obs, action, next_obs, rew, done)
            obs = next_obs
            env.render()

    # Plot Initial State Values
    plt.plot(init_state_values, label=label)

# plt.xlabel('Episode')
# plt.ylabel('$V(s_0)$')
# plt.legend(loc='lower right')
# plt.savefig('results/images/run_minigrid_{}.png'.format(SEED))
# plt.show()

import time

epi_return = 0
while not done:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    epi_return += rew
    env.render()
    time.sleep(0.01)

print('Episodic Return: ', epi_return)