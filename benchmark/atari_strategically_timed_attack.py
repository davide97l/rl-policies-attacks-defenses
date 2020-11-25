import os
import torch
import argparse
import numpy as np
from drl_attacks.strategically_timed_attack import strategically_timed_attack_collector
from utils import make_policy, make_img_adv_attack, make_atari_env_watch, make_victim_network
import time
import gym


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
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default="log/PongNoFrameskip-v4/dqn/policy.pth")
    parser.add_argument('--image_attack', type=str, default='fgm')  # fgm, cw
    parser.add_argument('--policy', type=str, default='dqn')  # dqn, a2c, ppo
    parser.add_argument('--perfect_attack', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--target_policy_path', type=str, default=None)  # log_2/PongNoFrameskip-v4/dqn/policy.pth
    parser.add_argument('--target_policy', type=str, default=None)  # dqn, a2c, ppo
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--no_softmax', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    args = parser.parse_known_args()[0]
    return args


def benchmark_adversarial_policy(args=get_args()):
    env = make_atari_env_watch(args)
    if args.save_video:
        log_path = os.path.join(args.logdir, args.task, args.policy, "strategically_timed_attack_eps-" + str(args.eps) +\
                                "_beta-" + str(args.beta) + "_" + args.target_policy)
        env = gym.wrappers.Monitor(env, log_path, force=True)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape: ", args.state_shape)
    print("Actions shape: ", args.action_shape)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # make policy
    policy = make_policy(args, args.policy, args.resume_path)
    # make target policy
    if args.target_policy is not None:
        victim_policy = make_policy(args, args.target_policy, args.target_policy_path)
        adv_net = make_victim_network(args, victim_policy)
    else:
        adv_net = make_victim_network(args, policy)
    # define observations adversarial attack
    obs_adv_atk, atk_type = make_img_adv_attack(args, adv_net, targeted=True)
    print("Attack type:", atk_type)

    # define adversarial collector
    collector = strategically_timed_attack_collector(policy, env, obs_adv_atk,
                                                     perfect_attack=args.perfect_attack,
                                                     softmax=False if args.no_softmax else True,
                                                     device=args.device)
    collector.beta = args.beta
    start_time = time.time()
    test_adversarial_policy = collector.collect(n_episode=args.test_num)
    print("Attack finished in %s seconds" % (time.time() - start_time))
    atk_freq_ = test_adversarial_policy['atk_rate(%)']
    reward = test_adversarial_policy['rew']
    n_attacks = test_adversarial_policy['n_atks']
    print("attack frequency =", atk_freq_, "| n_attacks =", n_attacks,
          "| n_succ_atks (%)", test_adversarial_policy['succ_atks(%)'],
          "| reward: ", reward)


if __name__ == '__main__':
    benchmark_adversarial_policy(get_args())
