import torch.nn as nn


# TODO Support this environment
class MinigridNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 7),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128)
        x = self.fc_layers(x)

        return x
