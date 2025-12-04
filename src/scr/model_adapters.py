import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_dim=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_size)
        # Init up to zeros so adapter initially does almost nothing
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    def forward(self, x):
        return self.up(self.act(self.down(x)))
