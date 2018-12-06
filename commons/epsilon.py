import numpy as np


def get_epsilon_decay_function(epsilon_start, epsilon_end, decay_duration):
    """
    Return a lambda function that returns `epsilon` when given a `frame_idx`. Decays over time for `decay_duration` steps.

    Parameters
    ----------
    epsilon_start : float
        Epsilon value to start with.
    epsilon_end : float
        Epsilon value to end with.
    decay_duration: int
        How long to decay the epsilon value for. After `decay_duration` steps, the epsilon value is fixed at `epsilon_end`.

    Returns
    -------
    get_epsilon_by_frame_idx
        A lambda function that returns `epsilon` when given a `frame_idx`. Decays over time for `decay_duration` steps.
    """
    return lambda frame_idx: epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * frame_idx / decay_duration)
