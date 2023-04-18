import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(91, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)       
        x = self.linear3(x)
        return x
