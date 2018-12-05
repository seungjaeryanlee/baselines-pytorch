from collections import deque
import random

import numpy as np
import torch


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return torch.cat(state), action, torch.cat(reward), torch.cat(next_state), torch.cat(done)

    def __len__(self):
        return len(self.buffer)
