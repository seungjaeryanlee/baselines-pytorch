import numpy as np


def get_epsilon_decay(start, final, steps):
    return lambda frame_idx: final + (start - final) * np.exp(-1. * frame_idx / steps)
