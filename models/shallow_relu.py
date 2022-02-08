import torch.nn as nn


class ShallowRelu(nn.Module):
    def __init__(self, n, input_dim, output_dim):
        super(ShallowRelu, self).__init__()
        self.hidden = nn.Linear(input_dim, n)
        self.relu = nn.ReLU()
        self.out = nn.Linear(n, output_dim)

    def forward(self, x):
        return self.out(self.relu(self.hidden(x)))
