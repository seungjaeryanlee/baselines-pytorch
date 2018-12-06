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
    args = parser.parse_args()

    return args
