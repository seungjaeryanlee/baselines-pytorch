import argparse

from agents import Agent
from commons import get_parser, set_seed, make_env


# Get hyperparameters from arguments
parser = get_parser()
args = parser.parse_args()

# Set Seed
set_seed(args.SEED)

# Create wrapped environment
env = make_env(args.ENV_ID)

# Create agent
agent = Agent(env, args)

# Train agent for args.NB_FRAMES
agent.train()

# Save agent
agent.save()

# Test agent
agent.test(render=False)
