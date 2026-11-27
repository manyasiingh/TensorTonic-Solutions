import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_size)
        self.r = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, out_features)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.r(self.l1(x))
        x = self.l2(x)
        return x
        pass
