import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import SubprocVectorEnv
from net.discrete_net import DQN
from drl_defenses.off_policy_trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from drl_defenses.adv_training import adversarial_training_collector
from atari_wrapper import wrap_deepmind
from utils import make_img_adv_attack, make_atari_env_watch, make_victim_network
import copy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps_test', type=float, default=0.005)
    parser.add_argument('--eps_train', type=float, default=0.01)
    parser.add_argument('--eps_train_final', type=float, default=0.01)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--target_update_freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step_per_epoch', type=int, default=10000)
    parser.add_argument('--collect_per_step', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--training_num', type=int, default=16)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--image_attack', type=str, default='fgm')  # fgm, cw, pgda
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--atk_freq', type=float, default=1.)
    parser.add_argument('--target_model_path', type=str, default=None,
                        help='model to base image adversarial attacks on')  # used for testing (i.e. log/PongNoFrameskip-v4/dqn/policy.pth)
    return parser.parse_args()


def make_atari_env(args):
    environment = wrap_deepmind(args.task, frame_stack=args.frames_stack)
    return environment


def test_dqn(args=get_args()):
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
                                  for _ in range(1)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    net = DQN(*args.state_shape,
              args.action_shape, args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = DQNPolicy(net, optim, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)

    if args.target_model_path:
        victim_policy = copy.deepcopy(policy)
        victim_policy.load_state_dict(torch.load(args.target_model_path))
        print("Loaded victim agent from: ", args.target_model_path)
    else:
        victim_policy = policy

    args.target_policy, args.policy = "dqn", "dqn"
    args.perfect_attack = False
    adv_net = make_victim_network(args, victim_policy)
    adv_atk, _ = make_img_adv_attack(args, adv_net, targeted=False)

    buffer = ReplayBuffer(args.buffer_size, ignore_obs_next=True)
    # collector
    train_collector = adversarial_training_collector(policy, train_envs, adv_atk, buffer,
                                                     atk_frequency=args.atk_freq, device=args.device)
    test_collector = adversarial_training_collector(policy, test_envs, adv_atk, buffer, atk_frequency=args.atk_freq,
                                                    test=True, device=args.device)
    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    writer = SummaryWriter(log_path)

    def save_fn(policy, policy_name='policy.pth'):
        torch.save(policy.state_dict(), os.path.join(log_path, policy_name))

    def stop_fn(x):
        return 0

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                  (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        writer.add_scalar('train/eps', eps, global_step=env_step)
        print("set eps =", policy.eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        assert args.target_model_path is not None
        print("Testing agent ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[args.test_num],
                                        render=args.render)
        pprint.pprint(result)

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * 4)
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer, test_in_train=False)

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_dqn(get_args())