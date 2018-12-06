import numpy as np


def get_epsilon_decay_function(e_start, e_end, decay_duration):
    """
    Return a lambda function that returns `epsilon` when given a `frame_idx`.
    Decays over time for `decay_duration` steps.

    Parameters
    ----------
    e_start : float
        Epsilon value to start with.
    e_end : float
        Epsilon value to end with.
    decay_duration: int
        How long to decay the epsilon value for. After `decay_duration` steps,
        the epsilon value is fixed at `e_end`.

    Returns
    -------
    get_epsilon_by_frame_idx
        A lambda function that returns `epsilon` when given a `frame_idx`.
        Decays over time for `decay_duration` steps.
    """
    return lambda frame_idx: e_end + \
        (e_start - e_end) * np.exp(-1. * frame_idx / decay_duration)
