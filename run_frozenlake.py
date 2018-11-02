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
NB_TRAINING_EPISODES = 1500
LEARNING_RATE = 0.1
BUFFER_LEARNING_RATE = 0.05
EPSILON = 0.1
BATCH_SIZE = 16

# For reproducibility
SEED = 0xc0ffee
random.seed(SEED)
np.random.seed(SEED)

# Setup Environment
env = get_frozenlake(seed=SEED)

# Setup Agents
online_agent = TabularQLearningAgent(env.observation_space, env.action_space,
                                lr=LEARNING_RATE, epsilon=EPSILON)


urb = UniformReplayBuffer(capacity=100)
uniform_agent = TabularBufferQAgent(env.observation_space,
                            env.action_space,
                            urb,
                            lr=BUFFER_LEARNING_RATE,
                            epsilon=EPSILON,
                            batch_size=BATCH_SIZE)

prb = PrioritizedReplayBuffer(capacity=100)
prioritized_agent = TabularBufferQAgent(env.observation_space,
                            env.action_space,
                            prb,
                            lr=BUFFER_LEARNING_RATE,
                            epsilon=EPSILON,
                            batch_size=BATCH_SIZE)

combined_replay_buffers = [
    # replay_buffer, batch_size
    (UniformReplayBuffer(capacity=1), 1),
    (UniformReplayBuffer(capacity=100), 15),
]
combined_agent = TabularEmsembleQAgent(env.observation_space,
                              env.action_space,
                              combined_replay_buffers,
                              lr=LEARNING_RATE,
                              epsilon=EPSILON)

ensemble_replay_buffers = [
    # replay_buffer, batch_size
    (PrioritizedReplayBuffer(capacity=10), 8),
    (PrioritizedReplayBuffer(capacity=100), 8),
]
ensemble_agent = TabularEmsembleQAgent(env.observation_space,
                              env.action_space,
                              ensemble_replay_buffers,
                              lr=LEARNING_RATE,
                              epsilon=EPSILON)

agents = [
    (online_agent, 'Online'),
    (uniform_agent, 'URB'),
    (prioritized_agent, 'PRB'),
    (combined_agent, 'CER'),
    (ensemble_agent, 'Ensemble'),
]

for agent, label in agents:

    # Train agent
    init_state_values = []
    for i in range(NB_TRAINING_EPISODES):
        obs = env.reset()
        init_state_value = agent.get_state_value(obs)
        init_state_values.append(init_state_value)
        if i % 100 == 99:
            print('Initial state value for episode {:5d}: {}'.format(i + 1, init_state_value))
        
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, rew, done, info = env.step(action)
            agent.learn(obs, action, next_obs, rew, done)
            obs = next_obs

    # Plot Initial State Values
    plt.plot(init_state_values, label=label)

plt.xlabel('Episode')
plt.ylabel('$V(s_0)$')
plt.legend(loc='lower right')
plt.savefig('results/images/run_frozenlake_{}.png'.format(SEED))
plt.show()
