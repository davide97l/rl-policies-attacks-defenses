from __future__ import division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnDQN(nn.Module):
    def __init__(self, num_channels, action_space):
        super(CnnDQN, self).__init__()
        
        self.num_actions = action_space.n
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.train()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                q_value = self.forward(state)
                #print(q_value)
                action  = torch.argmax(q_value, dim=1)[0]
            else:
                action = random.randrange(self.num_actions)
        return action