import torch
import pprint
import argparse
import numpy as np
from advertorch.attacks import *
import copy
from drl_attacks.uniform_attack import uniform_attack_collector
from atari_wrapper import wrap_deepmind
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, ReplayBuffer
from utils import NetAdapter, make_policy, make_img_adv_attack


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=3)  # only dqn
    parser.add_argument('--vf-coef', type=float, default=0.5)  # only a2c and ppo
    parser.add_argument('--ent-coef', type=float, default=0.01)  # only a2c and ppo
    parser.add_argument('--max-grad-norm', type=float, default=0.5)  # only a2c and ppo
    parser.add_argument('--target_update_freq', type=int, default=500)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default="log/PongNoFrameskip-v4/dqn/policy.pth")
    parser.add_argument('--image_attack', type=str, default='fgsm')  # [fgsm, cw]
    parser.add_argument('--policy', type=str, default='dqn')  # [dqn, a2c, ppo]
    parser.add_argument('--attack_freq', type=float, default=1.)
    parser.add_argument('--perfect_attack', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=0.3)  # fgsm and cw
    parser.add_argument('--iterations', type=int, default=100)  # only cw
    parser.add_argument('--target_policy_path', type=str, default=None)  # log_2/PongNoFrameskip-v4/dqn/policy.pth
    parser.add_argument('--target_policy', type=str, default=None)  # dqn, a2c, ppo
    parser.add_argument('--test_random', default=False, action='store_true')
    parser.add_argument('--test_normal', default=False, action='store_true')
    args = parser.parse_known_args()[0]
    return args


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)


def test_adversarial_policy(args=get_args()):
    env = make_atari_env_watch(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape: ", args.state_shape)
    print("Actions shape: ", args.action_shape)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # make policy
    policy, model = make_policy(args, args.policy, args.resume_path)
    # make target policy
    if args.target_policy is not None:
        _, model = make_policy(args, args.target_policy, args.target_policy_path)
    # define victim policy
    adv_net = NetAdapter(copy.deepcopy(model)).to(args.device)
    adv_net.eval()
    # define observations adversarial attack
    obs_adv_atk, atk_type = make_img_adv_attack(args, adv_net)
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
    test_adversarial_policy = collector.collect(n_episode=args.test_num, device=args.device)
    pprint.pprint(test_adversarial_policy)
