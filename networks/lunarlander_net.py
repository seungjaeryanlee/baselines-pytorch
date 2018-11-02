import torch.nn as nn


class LunarLanderNetwork(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(8, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, 4)
        )

    def forward(self, x):
        return self.layers(x)
