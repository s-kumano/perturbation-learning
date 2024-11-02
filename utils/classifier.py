import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TwoLayerNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, slope: float) -> None:
        super().__init__()

        self.linear = nn.Linear(in_dim, hidden_dim)
        nn.init.normal_(self.linear.weight, 0, 1 / math.sqrt(in_dim))
        nn.init.normal_(self.linear.bias, 0, 1)

        readout = torch.normal(0, 1 / math.sqrt(hidden_dim), (1, hidden_dim))
        self.register_buffer('readout', readout)

        self.slope = slope

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x.flatten(1))
        x = F.leaky_relu(x, self.slope)
        return (x @ self.readout.T).flatten()