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

        Parameters
        ----------
        state : torch.Tensor of torch.float32
            Has shape (1, L) (for feature-type states) or (1, C, H, W) (for image-type states).
        action : int
        reward : torch.Tensor of torch.float32
            Has shape (1, 1)
        next_state : torch.Tensor of torch.float32
            Has shape (1, L) (for feature-type states) or (1, C, H, W) (for image-type states).
        done : torch.Tensor of torch.float32
            Has shape (1, 1)
        """
        self.buffer.append((state, torch.LongTensor([action]), reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer.

        This function does not check if the buffer is bigger than the `batch_size`.

        Parameters
        ----------
        batch_size : int
            Size of the output batch. Should be at most the current size of the buffer.

        Returns
        -------
        batch: tuple of torch.Tensor
            A tuple of batches: (state_batch, action_batch, reward_batch, next_state_batch, done_batch).
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return torch.cat(state), torch.cat(action), torch.cat(reward), torch.cat(next_state), torch.cat(done)
