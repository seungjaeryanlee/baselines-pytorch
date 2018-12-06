import random
import numpy as np
import torch


def set_seed(env, seed):
    """
    Make the execution deterministic by setting seed to all nondeterministic parts.

    Parameters
    ----------
    env
        The environment that the agent will interact with. Could be either deterministic or stochastic.
    seed : int
        The seed to set all random number generators.
    """
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
