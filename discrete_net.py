import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    """Dense network
    Input: observations
    Output: actions"""
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class Actor(nn.Module):
    """Actor network
    Input: observations
    Output: actions"""
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h


class Critic(nn.Module):
    """Dense network
    Input: observations
    Output: scalar value"""
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, 1)

    def forward(self, s):
        logits, h = self.preprocess(s, None)
        logits = self.last(logits)
        return logits


class DQN(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    Reference paper: "Human-level control through deep reinforcement learning".
    """

    def __init__(self, d, h, w, action_shape, device='cpu'):
        super(DQN, self).__init__()
        self.device = device

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def conv2d_layers_size_out(size,
                                   kernel_size_1=8, stride_1=4,
                                   kernel_size_2=4, stride_2=2,
                                   kernel_size_3=3, stride_3=1):
            size = conv2d_size_out(size, kernel_size_1, stride_1)
            size = conv2d_size_out(size, kernel_size_2, stride_2)
            size = conv2d_size_out(size, kernel_size_3, stride_3)
            return size

        convw = conv2d_layers_size_out(w)
        convh = conv2d_layers_size_out(h)
        linear_input_size = convw * convh * 64

        self.net = nn.Sequential(
            nn.Conv2d(d, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.Linear(512, action_shape)
        )

    def forward(self, x, state=None, info={}):
        r"""x -> Q(x, \*)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x), state


class DQN2(nn.Module):
    """ConvNet architecture from https://arxiv.org/abs/1710.02298 (Rainbow)
    Input: observations
    Output: actions"""
    def __init__(self, h, w, action_shape, device='cpu'):
        super(DQN2, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,
                                                                kernel_size=8, stride=4),
                                                kernel_size=4, stride=2),
                                kernel_size=3, stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,
                                                                kernel_size=8, stride=4),
                                                kernel_size=4, stride=2),
                                kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, action_shape)

    def forward(self, x, state=None, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.reshape(x.size(0), -1))
        return self.head(x), state


class AntagonistNet(nn.Module):
    """Dense network for antagonist attack
    Input: observations
    Output: actions"""
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.actor = nn.Linear(128, np.prod(action_shape))
        self.atk_critic = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        output = self.model(s)
        logits = self.actor(output)
        p = self.sigmoid(self.atk_critic(output))
        return logits, p, state


if __name__ == '__main__':
    dims = [1, 84, 84, 4]
    s1 = torch.FloatTensor(np.random.uniform(size=dims))  # state
    dqn = DQN2(84, 84, 2)
    _, _ = dqn(s1)

