#!/usr/bin/env python3
"""
Run a specified agent on FrozenLakeDeterministic8x8-v0.
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import random

from agents import *
from common import get_running_mean, get_q_map
from envs import get_frozenlake
from replay import UniformReplayBuffer, PrioritizedReplayBuffer


# Hyperparameters
NB_TRAINING_EPISODES = 2000
NB_TEST_EPISODES = 100
LEARNING_RATE = 0.05
EPSILON = 0.1

# For reproducibility
SEED = 0xc0ffee
random.seed(SEED)
np.random.seed(SEED)

# Setup Agent & Environment
env = get_frozenlake(seed=SEED)

replay_buffers = [
    # replay_buffer, batch_size
    (PrioritizedReplayBuffer(capacity=10), 4),
    (PrioritizedReplayBuffer(capacity=1000), 12),
]
agent = TabularEmsembleQAgent(env.observation_space,
                              env.action_space,
                              replay_buffers,
                              lr=LEARNING_RATE,
                              epsilon=EPSILON)

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

# Plot Initial State Values
plt.xlabel('Episode')
plt.ylabel('$V(s_0)$')
plt.plot(init_state_values)
plt.savefig('results/images/run_frozenlake_with_online_q.png')
plt.show()

# Plot Smoothed Initial State Values
plt.xlabel('Episode')
plt.ylabel('$G$')
plt.plot(get_running_mean(init_state_values))
plt.savefig('results/images/run_frozenlake_with_online_q_smooth.png')
plt.show()

# Print Q-Map
print(get_q_map(env, agent))

# Test Agent
epi_returns = []
for i in range(NB_TEST_EPISODES):
    obs = env.reset()

    done = False
    epi_return = 0
    while not done:
        action = agent.get_action(obs, epsilon=0)
        obs, rew, done, info = env.step(action)
        epi_return += rew

    epi_returns.append(epi_return)

print('Average Episodic Return of ', np.mean(epi_returns))
