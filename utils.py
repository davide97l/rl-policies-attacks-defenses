import torch.nn as nn
import torch
from discrete_net import DQN, ConvNet, Actor, Critic
from tianshou.policy import DQNPolicy, A2CPolicy


class NetAdapter(nn.Module):
    """
    Tianshou models return (logits, state) while Advertorch models should return (logits).
    Hence, this class adapts Tianshou output to Advertorch output."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, s):
        return self.net(s)[0]


def make_dqn(args):
    net = DQN(*args.state_shape,
              args.action_shape, args.device).to(args.device)
    policy = DQNPolicy(net, None, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)
    policy.set_eps(args.eps_test)
    return policy, policy.model


def make_a2c(args):
    net = ConvNet(*args.state_shape, args.device).to(args.device)
    actor = Actor(net, args.action_shape).to(args.device)
    critic = Critic(net).to(args.device)
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor, critic, None, dist, args.gamma, vf_coef=args.vf_coef,
        ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm,
        target_update_freq=args.target_update_freq)
    return policy, policy.actor
