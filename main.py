#!/usr/bin/env python3
import torch

from agents import Agent
from commons import get_args, set_seed
from wrappers import make_env


# Get hyperparameters from arguments
args = get_args()

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
