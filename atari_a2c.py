import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import A2CPolicy
from tianshou.env import SubprocVectorEnv
from net.discrete_net import ConvNet, Actor, Critic
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

from atari_wrapper import wrap_deepmind


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step_per_epoch', type=int, default=100)
    parser.add_argument('--collect_per_step', type=int, default=1000)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_num', type=int, default=16)
    parser.add_argument('--test_num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=40)
    args = parser.parse_known_args()[0]
    return args


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)


def test_a2c(args=get_args()):
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape: ", args.state_shape)
    print("Actions shape: ", args.action_shape)
    # make environments
    train_envs = SubprocVectorEnv([lambda: make_atari_env(args)
                                   for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                                  for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    net = ConvNet(*args.state_shape, args.device).to(args.device)
    actor = Actor(net, args.action_shape).to(args.device)
    critic = Critic(net).to(args.device)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor, critic, optim, dist, args.gamma, vf_coef=args.vf_coef,
        ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)
    # collector
    train_collector = Collector(
        policy, train_envs,
        ReplayBuffer(args.buffer_size, ignore_obs_next=True))  # save memory
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'a2c')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        if env.env.spec.reward_threshold:
            return x >= env.spec.reward_threshold
        elif 'Pong' in args.task:
            return x >= 20

    # watch agent's performance
    def watch():
        print("Testing agent ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                                 for _ in range(args.test_num)])
        envs.seed(args.seed)
        collector = Collector(policy, envs)
        result = collector.collect(n_episode=args.test_num, render=args.render)
        pprint.pprint(result)

    if args.watch:
        watch()
        exit(0)

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer, test_in_train=False)

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_a2c(get_args())
