import torch.nn as nn


class ShallowRelu(nn.Module):
    def __init__(self, n):
        super(ShallowRelu, self).__init__()
        self.shallow_relu = nn.Sequential(
            nn.Linear(1, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )

    def forward(self, x):
        return self.shallow_relu(x)
