import os
import torch
import argparse
import numpy as np
from drl_attacks.adversarial_policy_attack import adversarial_policy_attack_collector
from utils import make_policy, make_img_adv_attack, make_atari_env_watch, make_victim_network


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
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default="log/PongNoFrameskip-v4/dqn/policy.pth")
    parser.add_argument('--image_attack', type=str, default='fgm')  # fgm, cw
    parser.add_argument('--policy', type=str, default='dqn')  # dqn, a2c, ppo
    parser.add_argument('--perfect_attack', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--target_policy_path', type=str, default=None)  # log_2/PongNoFrameskip-v4/dqn/policy.pth
    parser.add_argument('--target_policy', type=str, default=None)  # dqn, a2c, ppo
    parser.add_argument('--min', type=float, default=0.2)
    parser.add_argument('--max', type=float, default=0.3)
    parser.add_argument('--steps', type=int, default=11)
    parser.add_argument('--no_softmax', default=False, action='store_true')
    parser.add_argument('--adv_policy', type=str, default=None)  # dqn, a2c, ppo
    parser.add_argument('--adv_policy_path', type=str, default=None)  # log_adv_policy/PongNoFrameskip-v4/dqn/policy.pth
    args = parser.parse_known_args()[0]
    return args


def benchmark_adversarial_policy(args=get_args()):
    env = make_atari_env_watch(args)
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
    transferability_type = ""
    if args.target_policy is not None:
        victim_policy = make_policy(args, args.target_policy, args.target_policy_path)
        transferability_type = "_transf_" + str(args.target_policy)
        adv_net = make_victim_network(args, victim_policy)
    else:
        adv_net = make_victim_network(args, policy)
    # define observations adversarial attack
    obs_adv_atk, atk_type = make_img_adv_attack(args, adv_net, targeted=True)
    print("Attack type:", atk_type)

    # define adversarial policy
    adv_policy = None
    if args.adv_policy is not None:
        adv_policy = make_policy(args, args.adv_policy, args.adv_policy_path)
    # define adversarial collector
    collector = adversarial_policy_attack_collector(policy, env, obs_adv_atk,
                                                    perfect_attack=args.perfect_attack,
                                                    softmax=False if args.no_softmax else True,
                                                    device=args.device, adv_policy=adv_policy)
    beta = np.linspace(args.min, args.max, args.steps, endpoint=True)
    atk_freq = []
    n_attacks = []
    rewards = []
    for b in beta:
        collector.beta = b
        test_adversarial_policy = collector.collect(n_episode=args.test_num)
        rewards.append(test_adversarial_policy['rew'])
        atk_freq.append(test_adversarial_policy['atk_rate(%)'])
        n_attacks.append(test_adversarial_policy['n_atks'])
        print("attack frequency =", atk_freq[-1], "| n_attacks =", n_attacks[-1],
              "| n_succ_atks (%)", test_adversarial_policy['succ_atks(%)'],
              "| reward: ", rewards[-1])
        # pprint.pprint(test_adversarial_policy)
    log_path = os.path.join(args.logdir, args.task, args.policy,
                            "adversarial_policy_attack_" + atk_type + transferability_type + ".npy")

    with open(log_path, 'wb') as f:
        np.save(f, atk_freq)
        np.save(f, n_attacks)
        np.save(f, rewards)
    print("Results saved to", log_path)


if __name__ == '__main__':
    benchmark_adversarial_policy(get_args())
