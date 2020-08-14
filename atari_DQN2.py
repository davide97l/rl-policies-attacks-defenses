import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import VectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

from discrete_net import DQN
from atari_wrapper import *

#from atari import create_atari_environment, preprocess_fn


"""def get_args():
    class Args(object):
        def __init__(self,):
            self.task = 'PongNoFrameskip-v4'
            self.seed = 0
            self.eps_test = 0.01
            self.eps_train = 0.5
            self.eps_train_final = 0.05
            self.buffer_size = 100000
            self.lr = 0.0001
            self.gamma = 0.99
            self.n_step = 3
            self.target_update_freq = 100
            self.epoch = 100
            self.step_per_epoch = 10000
            self.collect_per_step = 10
            self.batch_size = 64
            self.training_num = 8
            self.test_num = 100
            self.logdir = 'log'
            self.render = 0.
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.cuda_device = 0  # int
            self.frames_stack = 4
            self.resume_path = "log/PongNoFrameskip-v4/dqn/policy.pth"
            self.watch = True
    args = Args()
    return args"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps_test', type=float, default=0.01)
    parser.add_argument('--eps_train', type=float, default=0.5)
    parser.add_argument('--eps_train_final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step_per_epoch', type=int, default=10000)
    parser.add_argument('--collect_per_step', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_num', type=int, default=8)
    parser.add_argument('--test_num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default=None)  # "log/PongNoFrameskip-v4/dqn/policy.pth"
    parser.add_argument('--watch', default=False, action='store_true',
                        help='no training, watch the play of pre-trained models')
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    if args.cuda_device is not None:
        print("Available devices: ", torch.cuda.device_count())
        device_name = 'cuda:' + str(args.cuda_device)
        args.device = torch.device(device_name)
        print("Selected device: ", args.device)

    def make_atari_env(env):
        env = make_atari(env, frame_stack=args.frames_stack)
        return env

    def make_atari_env_watch(env):
        env = make_atari(env, frame_stack=args.frames_stack, episode_life=False, clip_rewards=False)
        return env

    preprocess_fn = None

    env = make_atari_env(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    print("Observations shape: ", args.state_shape)  # should be H x W x N_FRAMES
    print("Actions shape: ", args.action_shape)
    # make environments
    train_envs = SubprocVectorEnv([
        lambda: make_atari_env(args.task)
        for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([
        lambda: make_atari_env(args.task)
        for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    net = DQN(
        args.state_shape[0], args.state_shape[1], args.state_shape[2] if len(args.state_shape) > 2 else 1,
        args.action_shape, args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        use_target_network=args.target_update_freq > 0,
        target_update_freq=args.target_update_freq)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size),
        preprocess_fn=preprocess_fn)
    test_collector = Collector(policy, test_envs, preprocess_fn=preprocess_fn)
    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * 4)
    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')

    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        if env.env.spec.reward_threshold:
            return x >= env.spec.reward_threshold
        else:
            return False

    def train_fn(x):
        if x <= int(0.1 * args.epoch):
            policy.set_eps(args.eps_train)
        elif x <= int(0.5 * args.epoch):
            eps = args.eps_train - (x - 0.1 * args.epoch) / \
                  (0.4 * args.epoch) * (0.8 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(args.eps_train_final)
        print("eps: ", policy.eps)

    def test_fn(x):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Testing agent...")
        env = make_atari_env_watch(args.task)
        collector = Collector(policy, env, preprocess_fn=preprocess_fn)
        result = collector.collect(n_episode=args.test_num, render=args.render)
        print("Final reward: ", str(result["rew"]), " length: ", str(result["len"]), " episodes: ", args.test_num)
        collector.close()

    if args.watch is True:
        watch()
        exit(0)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer)

    train_collector.close()
    test_collector.close()
    pprint.pprint(result)
    print("Agent saved to: ", os.path.join(log_path, 'policy.pth'))

    if __name__ == '__main__':
        watch()


if __name__ == '__main__':
    test_dqn(get_args())
