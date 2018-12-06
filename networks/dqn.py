import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        """
        A simple feedforward deep Q-network with two hidden layers with 128
        nodes each. Can be used with environments with feature-type states.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)
