import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FullyConnectedNet, self).__init__()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.output = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)
