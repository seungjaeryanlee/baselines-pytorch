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
from envs import get_lunarlander
from common import get_running_mean
from networks import LunarLanderNetwork
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.optim as optim


# Hyperparameters
NB_TRAINING_EPISODES = 200
LEARNING_RATE = 5e-4
EPSILON = 0.1

# For reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

SEED = 0xc0ffee
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Setup Environment
env = get_lunarlander(SEED)

# Setup Agents
q_network = LunarLanderNetwork(64, 64)
optimizer = optim.RMSprop(q_network.parameters(), lr=LEARNING_RATE)
online_agent = ApproximateQLearningAgent(q_network,
                                         optimizer,
                                         env.observation_space,
                                         env.action_space,
                                         epsilon=0.1,
                                         gamma=0.99)

agents = [
    (online_agent, 'Online'),
]


for agent, label in agents:

    # Train agent
    epi_returns = []
    for i in range(NB_TRAINING_EPISODES):
        obs = env.reset()    
        epi_return = 0
        
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, rew, done, info = env.step(action)
            epi_return +=  rew
            agent.learn(obs, action, next_obs, rew, done)
            obs = next_obs

        print('Episode | {:5d} | Return | {:5.2f}'.format(i + 1, epi_return))
        epi_returns.append(epi_return)

    # Plot episodic return
    plt.plot(get_running_mean(epi_returns), label=label)

plt.xlabel('Episode')
plt.ylabel('$G$')
plt.savefig('results/images/run_lunarlander_{}.png'.format(SEED))
plt.show()
