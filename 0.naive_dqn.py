#!/usr/bin/env python3
import argparse
import random
import time

import gym
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from agents import NaiveDQNAgent
from commons import set_seed, get_epsilon_decay
from replays import UniformReplayBuffer
from networks import DQN
from wrappers import TorchWrapper


# Parse Arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--seed', action='store', dest='SEED', default=1, type=int)
parser.add_argument('-n', '--frames', action='store', dest='NB_FRAMES', default=10000, type=int)
parser.add_argument('-b', '--batch', action='store', dest='BATCH_SIZE', default=32, type=int)
parser.add_argument('-d', '--discount', action='store', dest='DISCOUNT', default=0.99, type=float)
parser.add_argument('-u', '--update', action='store', dest='TARGET_UPDATE_STEPS', default=100, type=int)
parser.add_argument('-l', '--lr', action='store', dest='LEARNING_RATE', default=1e-3, type=int)
args = parser.parse_args()

# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup Environment
env_id = 'CartPole-v0'
env = gym.make(env_id)
env = TorchWrapper(env)

# Set Seed
set_seed(env, args.SEED)

# Setup Agent
model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
agent = NaiveDQNAgent(env, model, optimizer, device, discount=args.DISCOUNT)

# Setup Epsilon Decay
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_STEPS = 500
epsilon_by_frame = get_epsilon_decay(EPSILON_START, EPSILON_FINAL, EPSILON_DECAY_STEPS)

def train(nb_frames):
    episode_reward = 0
    nb_episode = 0
    loss = 0
    state = env.reset()
    writer = SummaryWriter('runs/{}/Naive/{}/{}/{}/{}/{}'.format(env_id, args.SEED, args.NB_FRAMES, args.BATCH_SIZE, args.DISCOUNT, args.TARGET_UPDATE_STEPS))
    for frame_idx in range(1, nb_frames + 1):
        time_start = time.time()
        epsilon = epsilon_by_frame(frame_idx)
        action = agent.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        writer.add_scalar('data/rewards', reward, frame_idx)

        state = next_state
        episode_reward += reward

        if done:
            print('Frame {:5d}/{:5d} Reward {:3d} Loss {:2.4f}'.format(frame_idx + 1, nb_frames, int(episode_reward), loss))
            writer.add_scalar('data/episode_rewards', episode_reward, nb_episode)
            state = env.reset()
            episode_reward = 0
            nb_episode += 1

        loss = agent.train(state, action, reward, next_state, done)
        writer.add_scalar('data/losses', loss.item(), frame_idx)

        time_end = time.time()
        writer.add_scalar('data/time', time_end - time_start, frame_idx)

    writer.close()


if __name__ == '__main__':
    train(args.NB_FRAMES)
