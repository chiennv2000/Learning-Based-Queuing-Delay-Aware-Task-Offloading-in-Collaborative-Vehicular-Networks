import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, output_size),
        )
    def forward(self, x):
        return self.linear(x)