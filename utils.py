import torch.nn as nn
import torch
from net.discrete_net import DQN, ConvNet, Actor, Critic
from tianshou.policy import DQNPolicy, A2CPolicy
from advertorch.attacks import *


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
    """Make a DQN policy
    :return: policy, actor network"""
    net = DQN(*args.state_shape,
              args.action_shape, args.device).to(args.device)
    policy = DQNPolicy(net, None, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)
    policy.set_eps(0)
    return policy, policy.model


def make_a2c(args):
    """Make a A2C policy
    :return: policy, actor network"""
    net = ConvNet(*args.state_shape, args.device).to(args.device)
    actor = Actor(net, args.action_shape).to(args.device)
    critic = Critic(net).to(args.device)
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor, critic, None, dist, args.gamma, vf_coef=args.vf_coef,
        ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm,
        target_update_freq=args.target_update_freq)
    return policy, policy.actor


def make_policy(args, policy_type, resume_path):
    """Make a 'policy_type' policy
    :return: policy, actor network"""
    assert policy_type in ["dqn", "a2c", "ppo"]
    policy, model = None, None
    if policy_type == "dqn":
        policy, model = make_dqn(args)
    if policy_type == "a2c":
        policy, model = make_a2c(args)
    if resume_path:
        policy.load_state_dict(torch.load(resume_path))
        print("Loaded agent from: ", resume_path)
    policy.eval()
    policy.set_eps(0.005)
    return policy, model


def make_img_adv_attack(args, adv_net, min_pixel=0., max_pixel=255., targeted=False):
    assert args.image_attack in ["fgm", "cw"] or args.perfect_attack
    obs_adv_atk, atk_type = None, None
    if args.perfect_attack:
        atk_type = "perfect_attack"
    elif args.image_attack == 'fgm':
        obs_adv_atk = GradientSignAttack(adv_net, eps=args.eps*max_pixel,
                                     clip_min=min_pixel, clip_max=max_pixel, targeted=targeted)
        atk_type = "fgm_eps_" + str(args.eps)
    elif args.image_attack == 'cw':
        obs_adv_atk = CarliniWagnerL2Attack(adv_net, args.action_shape, confidence=0.1,
                                            max_iterations=args.iterations,
                                            clip_min=min_pixel, clip_max=max_pixel, targeted=targeted)
        atk_type = "cw_iter_" + str(args.iterations)
    return obs_adv_atk, atk_type
