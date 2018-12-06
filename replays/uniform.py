from collections import deque
import random

import numpy as np
import torch


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new interaction / experience to the replay buffer.

        TODO Implement

        Parameters
        ----------
        state : list or np.array
        action : int
        reward : float
        next_state : list or np.array
        done : bool
        """
        pass

    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer.

        This function does not check if the buffer is bigger than the `batch_size`.

        TODO Implement

        Parameters
        ----------
        batch_size : int
            Size of the output batch. Should be at most the current size of the buffer.

        Returns
        -------
        batch: tuple of torch.Tensor
            A tuple of batches: (state_batch, action_batch, reward_batch, next_state_batch, done_batch).
        """
        pass
