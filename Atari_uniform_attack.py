import os
import torch
import pprint
import argparse
import numpy as np
from advertorch.attacks import *
import copy
from drl_attacks.uniform_attack import uniform_attack_collector
from atari_wrapper import wrap_deepmind
from discrete_net import DQN, ConvNet, Actor, Critic
from tianshou.policy import DQNPolicy, A2CPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, ReplayBuffer
from utils import NetAdapter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps_test', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=3)  # only dqn
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--step_per_epoch', type=int, default=10000)
    parser.add_argument('--collect_per_step', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default="log/PongNoFrameskip-v4/dqn/policy.pth")
    parser.add_argument('--image_attack', type=str, default='fgsm')  # fgsm, cw
    parser.add_argument('--policy', type=str, default='a2c')  # dqn, a2c, ppo
    parser.add_argument('--attack_freq', type=float, default=1.)
    parser.add_argument('--perfect_attack', default=True, action='store_true')
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--test_random', default=True, action='store_false')
    parser.add_argument('--test_normal', default=True, action='store_false')
    parser.add_argument('--repeat-per-collect', type=int, default=1)  # only a2c and ppo
    parser.add_argument('--vf-coef', type=float, default=0.5)  # only a2c and ppo
    parser.add_argument('--ent-coef', type=float, default=0.01)  # only a2c and ppo
    parser.add_argument('--max-grad-norm', type=float, default=40)  # only a2c and ppo
    args = parser.parse_known_args()[0]
    return args


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)


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


def test_adversarial_policy(args=get_args()):
    image_attack = ["fgsm", "cw"]
    victim_policy = ["dqn", "a2c", "ppo"]
    assert args.image_attack in image_attack or args.perfect_attack
    assert args.policy in victim_policy
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape: ", args.state_shape)
    print("Actions shape: ", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.policy == "dqn":
        policy, model = make_dqn(args)
    if args.policy == "a2c":
        policy, model = make_a2c(args)
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)
    policy.eval()
    # define victim policy
    adv_net = NetAdapter(copy.deepcopy(model)).to(args.device)
    adv_net.eval()
    # define observations adversarial attack
    obs_adv_atk = None
    if args.image_attack == 'fgsm':
        obs_adv_atk = GradientSignAttack(adv_net, eps=args.eps)
    if args.image_attack == 'cw':
        obs_adv_atk = CarliniWagnerL2Attack(adv_net, args.action_shape, confidence=0.1,
                                            max_iterations=100)  # define adversarial attack
    # make envs
    envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                             for _ in range(args.test_num)])
    envs.seed(args.seed)
    # define collector
    collector = Collector(policy, envs)
    if args.test_normal:
        print("Testing agent policy...")
        test_normal_policy = collector.collect(n_episode=args.test_num)
        pprint.pprint(test_normal_policy)
    if args.test_random:
        print("Testing random policy...")
        test_random_policy = collector.collect(n_episode=args.test_num, random=True)
        pprint.pprint(test_random_policy)

    print("Testing adversarial policy...")
    # define adversarial collector
    collector = uniform_attack_collector(policy, env, obs_adv_atk,
                                         atk_frequency=args.attack_freq,
                                         perfect_attack=args.perfect_attack)
    test_adversarial_policy = collector.collect(n_episode=args.test_num)
    pprint.pprint(test_adversarial_policy)
