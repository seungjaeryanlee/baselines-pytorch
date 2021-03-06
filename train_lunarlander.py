#!/usr/bin/env python3
import torch

from agents import Agent
from commons import get_args, set_seed
from wrappers import make_env


# Get hyperparameters from dictionary
args_dict = {
    'ENV_ID': 'LunarLander-v2',
    'SEED': 1,
    'NB_FRAMES': 100000,
    'BATCH_SIZE': 256,
    'DISCOUNT': 0.99,
    'TARGET_UPDATE_STEPS': 100,
    'LEARNING_RATE': 5e-4,
    'REPLAY_BUFFER_SIZE': 100000,
    'MIN_REPLAY_BUFFER_SIZE': 1000,
    'EPSILON_START': 1,
    'EPSILON_END': 0.1,
    'EPSILON_DECAY_DURATION': 50000,
}
# Allow changing hyperparameters from command-line arguments
args = get_args(default_args=args_dict)


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
agent.test(render=False)
