import argparse


def get_args(description=''):
    """
    Parse arguments with argument parser for and return hyperparameters as a dictionary.

    TODO This seems to be environment-dependent.

    Parameters
    ----------
    description: str
        Description for the argument parser. Defaults to an empty string.

    Returns
    -------
    args : dict
        Dictionary containing hyperparameter options specified by user or set by default.
    """
    parser = argparse.ArgumentParser(description)
    parser.add_argument('-e', '--envid', action='store', dest='ENV_ID', default='CartPole-v0', type=str)
    parser.add_argument('-s', '--seed', action='store', dest='SEED', default=1, type=int)
    parser.add_argument('-n', '--frames', action='store', dest='NB_FRAMES', default=10000, type=int)
    parser.add_argument('-b', '--batch', action='store', dest='BATCH_SIZE', default=32, type=int)
    parser.add_argument('-d', '--discount', action='store', dest='DISCOUNT', default=0.99, type=float)
    parser.add_argument('-u', '--update', action='store', dest='TARGET_UPDATE_STEPS', default=100, type=int)
    parser.add_argument('-l', '--lr', action='store', dest='LEARNING_RATE', default=1e-3, type=int)
    args = parser.parse_args()

    return args
