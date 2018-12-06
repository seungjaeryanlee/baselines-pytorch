#!/usr/bin/env python3
import argparse
import gym
import numpy as np
import random
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from agents import PERAgent
from commons import set_seed
from replays import PrioritizedReplayBuffer
from networks import DQN


# Parse Arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--seed', action='store', dest='seed', default=1, type=int)
parser.add_argument('-n', '--frames', action='store', dest='nb_frames', default=10000, type=int)
parser.add_argument('-b', '--batch', action='store', dest='batch_size', default=32, type=int)
parser.add_argument('-d', '--discount', action='store', dest='discount', default=0.99, type=float)
parser.add_argument('-u', '--update', action='store', dest='target_update_steps', default=100, type=int)
parser.add_argument('-l', '--lr', action='store', dest='lr', default=1e-3, type=int)
args = parser.parse_args()

# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
SEED = args.seed
NB_FRAMES = args.nb_frames
BATCH_SIZE = args.batch_size
DISCOUNT   = args.discount
TARGET_UPDATE_STEPS = args.target_update_steps
LEARNING_RATE = args.lr

# Setup Environment
env_id = 'CartPole-v0'
env = gym.make(env_id)

# Set Seed
set_seed(env, SEED)

# Setup Agent
current_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(current_net.parameters(), lr=LEARNING_RATE)
replay_buffer = PrioritizedReplayBuffer(1000)
agent = PERAgent(env, current_net, target_net, replay_buffer, optimizer, discount=0.99)

# Setup Epsilon Decay
# TODO Modularize
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = NB_FRAMES // 2
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

# Setup Beta Growth
beta_start = 0.4
beta_frames = NB_FRAMES
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

def train(nb_frames):
    episode_reward = 0
    nb_episode = 0
    loss = 0
    state = env.reset()
    writer = SummaryWriter('runs/{}/PER/{}/{}/{}/{}/{}'.format(env_id, SEED, NB_FRAMES, BATCH_SIZE, DISCOUNT, TARGET_UPDATE_STEPS))
    for frame_idx in range(1, nb_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        beta = beta_by_frame(frame_idx)
        action = agent.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        writer.add_scalar('data/rewards', reward, frame_idx)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            print('Frame {:5d}/{:5d} Reward {:3d} Loss {:2.4f}'.format(frame_idx + 1, nb_frames, int(episode_reward), loss))
            writer.add_scalar('data/episode_rewards', episode_reward, nb_episode)
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
