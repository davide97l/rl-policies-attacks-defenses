import torch.nn as nn
import os
import torch


class NetAdapter(nn.Module):
    """
    Tianshou models return (logits, state) while Advertorch models should return (logits).
    Thus this class adapts Tianshou output to Advertorch output
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, s):
        return self.net(s)[0]


def save_model(net, model_name, save_path="models"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_path = os.path.join(save_path, model_name + ".tar")
    torch.save({
        'model': net.state_dict(),
    }, os.path.join(full_path))
    print("model saved at " + full_path)


def load_model(net, model_name, save_path="models"):
    full_path = os.path.join(save_path, model_name + ".tar")
    checkpoint = torch.load(full_path)
    net.load_state_dict(checkpoint['model'])
    return net
