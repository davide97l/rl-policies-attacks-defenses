import torch.nn as nn
import torch
from net.discrete_net import DQN, ConvNet, Actor, Critic
from tianshou.policy import DQNPolicy, A2CPolicy
from advertorch.attacks import *
from atari_wrapper import wrap_deepmind
import os
import copy


class TianshouNetAdapter(nn.Module):
    """
    Tianshou models return (logits, state) while Advertorch models should return (logits).
    Hence, this class adapts Tianshou output to Advertorch output."""
    def __init__(self, policy):
        super().__init__()
        self.net = policy.model

    def forward(self, s):
        return self.net(s)[0]


class A2CPPONetAdapter(nn.Module):
    """
    Adapt the output of A2C-PPO models to Advertorch required output (logits)."""
    def __init__(self, policy):
        super().__init__()
        self.net = policy.base
        self.rnn_hxs = torch.zeros(1, policy.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)
        self.dist = policy.dist

    def forward(self, s):
        value, actor_features, rnn_hxs = self.net(s, self.rnn_hxs, self.masks)
        self.rnn_hxs = rnn_hxs
        dist = self.dist(actor_features)
        return dist.logits


def make_dqn(args):
    """Make a DQN policy
    :return: policy"""
    net = DQN(*args.state_shape,
              args.action_shape, args.device).to(args.device)
    policy = DQNPolicy(net, None, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)
    policy.set_eps(0)
    return policy


def make_a2c(args, resume_path):
    """Make a A2C policy
    :return: policy"""
    actor_critic, _ = torch.load(resume_path)
    actor_critic.to(args.device).init(args.device)
    return actor_critic


def make_ppo(args, resume_path):
    """Make a PPO policy
    :return: policy"""
    actor_critic, _ = torch.load(resume_path)
    actor_critic.to(args.device).init(args.device)
    return actor_critic


def make_policy(args, policy_type, resume_path):
    """Make a 'policy_type' policy
    :return: policy"""
    assert policy_type in ["dqn", "a2c", "ppo"]
    policy = None
    if policy_type == "dqn":
        policy = make_dqn(args)
    if policy_type == "a2c":
        assert resume_path is not None
        policy = make_a2c(args, resume_path)
    if policy_type == "ppo":
        assert resume_path is not None
        policy = make_ppo(args, resume_path)
    if resume_path:
        if policy_type == "dqn":
            policy.load_state_dict(torch.load(resume_path))
        print("Loaded agent from: ", resume_path)
    policy.eval()
    if hasattr(policy, 'eps'):
        policy.set_eps(0.005)
    return policy


def make_img_adv_attack(args, adv_net, min_pixel=0., max_pixel=255., targeted=False):
    assert args.image_attack in ["fgm", "cw", "pgda"] or args.perfect_attack
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
    elif args.image_attack == "pgda":
        obs_adv_atk = PGDAttack(adv_net, eps=args.eps*max_pixel, targeted=targeted,
                                clip_min=min_pixel, clip_max=max_pixel, nb_iter=args.iterations,
                                eps_iter=args.eps*max_pixel)
    return obs_adv_atk, atk_type


def make_victim_network(args, policy):
    if args.target_policy is None:
        if args.policy == 'dqn':
            adv_net = TianshouNetAdapter(copy.deepcopy(policy)).to(args.device)
        if args.policy in ['a2c', 'ppo']:
            adv_net = A2CPPONetAdapter(copy.deepcopy(policy)).to(args.device)
    elif args.target_policy == 'dqn':
        adv_net = TianshouNetAdapter(copy.deepcopy(policy)).to(args.device)
    elif args.target_policy in ['a2c', 'ppo']:
        adv_net = A2CPPONetAdapter(copy.deepcopy(policy)).to(args.device)
    adv_net.eval()
    return adv_net


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)
