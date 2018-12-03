#!/usr/bin/env python3
import gym
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from agents import PERAgent
from replays import PrioritizedReplayBuffer
from networks import DQN


# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NB_FRAMES = 10000
BATCH_SIZE = 32
DISCOUNT   = 0.99
TARGET_UPDATE_STEPS = 100

# Setup Environment
env = gym.make('CartPole-v0')

# Setup Agent
current_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(current_net.parameters())
replay_buffer = PrioritizedReplayBuffer(1000)
agent = PERAgent(env, current_net, target_net, replay_buffer, optimizer, discount=0.99)

# Setup Epsilon Decay
# TODO Modularize
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

# Setup Beta Growth
beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

def train(nb_frames):
    episode_reward = 0
    nb_episode = 0
    loss = 0
    state = env.reset()
    writer = SummaryWriter(comment='-PER')
    for frame_idx in range(1, nb_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        beta = beta_by_frame(frame_idx)
        action = agent.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            print('Frame {:5d}/{:5d} Reward {:3d} Loss {:2.4f}'.format(frame_idx + 1, nb_frames, int(episode_reward), loss))
            writer.add_scalar('data/rewards', episode_reward, nb_episode)
            state = env.reset()
            episode_reward = 0
            nb_episode += 1

        if len(replay_buffer) > BATCH_SIZE:
            loss = agent.train(BATCH_SIZE, beta)
            writer.add_scalar('data/losses', loss.item(), frame_idx)
        
        if (frame_idx + 1) % TARGET_UPDATE_STEPS == 0:
            agent.update_target()

    writer.close()


if __name__ == '__main__':
    train(NB_FRAMES)
