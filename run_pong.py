#!/usr/bin/env python3
from types import SimpleNamespace

import torch

from agents import Agent
from commons import get_args, set_seed
from wrappers import make_env


# Get hyperparameters from dictionary
args_dict = {
    'ENV_ID': 'PongNoFrameskip-v4',
    'SEED': 1,
    'NB_FRAMES': 1000000,
    'BATCH_SIZE': 32,
    'DISCOUNT': 0.99,
    'TARGET_UPDATE_STEPS': 1000,
    'LEARNING_RATE': 1e-4,
    'REPLAY_BUFFER_SIZE': 100000,
    'MIN_REPLAY_BUFFER_SIZE': 10000,
    'EPSILON_START': 1,
    'EPSILON_END': 0.02,
    'EPSILON_DECAY_DURATION': 100000,
}
args = SimpleNamespace(**args_dict)


# Create wrapped environment
env = make_env(args.ENV_ID)

# Set Seed
set_seed(env, args.SEED)

# GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create agent
agent = Agent(env, device, args)

# Train agent for args.NB_FRAMES
agent.train()

# Save agent
agent.save()

# Test agent
agent.test()
