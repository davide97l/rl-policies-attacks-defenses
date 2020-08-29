import os
import torch
import argparse
import numpy as np
from advertorch.attacks import *
import copy
from drl_attacks.uniform_attack import uniform_attack_collector
from atari_wrapper import wrap_deepmind
from utils import NetAdapter, make_dqn, make_a2c


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps_test', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=3)  # only dqn
    parser.add_argument('--vf-coef', type=float, default=0.5)  # only a2c and ppo
    parser.add_argument('--ent-coef', type=float, default=0.01)  # only a2c and ppo
    parser.add_argument('--max-grad-norm', type=float, default=0.5)  # only a2c and ppo
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default="log/PongNoFrameskip-v4/dqn/policy.pth")
    parser.add_argument('--image_attack', type=str, default='fgsm')  # fgsm, cw
    parser.add_argument('--policy', type=str, default='dqn')  # dqn, a2c, ppo
    parser.add_argument('--perfect_attack', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    args = parser.parse_known_args()[0]
    return args


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)


def benchmark_adversarial_policy(args=get_args()):
    image_attack = ["fgsm", "cw"]
    victim_policy = ["dqn", "a2c", "ppo"]
    assert args.image_attack in image_attack or args.perfect_attack
    assert args.policy in victim_policy
    env = make_atari_env_watch(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape: ", args.state_shape)
    print("Actions shape: ", args.action_shape)
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
    if args.perfect_attack:
        atk_type = "perfect_attack"
    elif args.image_attack == 'fgsm':
        obs_adv_atk = GradientSignAttack(adv_net, eps=args.eps)
        atk_type = "fgsm_eps_" + str(args.eps)
    elif args.image_attack == 'cw':
        obs_adv_atk = CarliniWagnerL2Attack(adv_net, args.action_shape, confidence=0.1,
                                            max_iterations=args.iterations)  # define adversarial attack
        atk_type = "cw_it_" + str(args.iterations)

    collector = uniform_attack_collector(policy, env, obs_adv_atk,
                                         perfect_attack=args.perfect_attack)
    atk_freq = np.linspace(0, 1, 21, endpoint=True)
    n_attacks = []
    rewards = []
    for f in atk_freq:
        collector.change_frequency(f)
        test_adversarial_policy = collector.collect(n_episode=args.test_num)
        rewards.append(test_adversarial_policy['rew'])
        n_attacks.append(test_adversarial_policy['n_atks'])
        print("attack frequency =", f, "| n_attacks =", n_attacks[-1], "| reward: ", rewards[-1])
        # pprint.pprint(test_adversarial_policy)
    log_path = os.path.join(args.logdir, args.task, args.policy, "uniform_attack_" + atk_type + ".npy")
    with open(log_path, 'wb') as f:
        np.save(f, atk_freq)
        np.save(f, n_attacks)
        np.save(f, rewards)
    print("Results saved to", log_path)


if __name__ == '__main__':
    benchmark_adversarial_policy(get_args())