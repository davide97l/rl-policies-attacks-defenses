import torch.nn as nn
import os
import torch


class NetAdapter(nn.Module):
    """
    Tianshou models return (logits, state) while Advertorch models should return (logits).
    Hence, this class adapts Tianshou output to Advertorch output."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, s):
        return self.net(s)[0]
