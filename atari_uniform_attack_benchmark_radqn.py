import os
import torch
import argparse
import numpy as np
from drl_attacks.uniform_attack import uniform_attack_collector
from utils import make_policy, make_img_adv_attack, make_atari_env_watch, make_victim_network
from defended_models.radial_dqn.environment import atari_env
from defended_models.radial_dqn.utils import read_config
from defended_models.radial_dqn.model import CnnDQN
from defended_models.radial_dqn.train import train
from utils import RadialDQNPolicyAdapter


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
    parser.add_argument('--resume_path', type=str, default="log/PongNoFrameskip-v4/radqn/policy.pth")
    parser.add_argument('--image_attack', type=str, default='fgm')  # fgsm, cw
    parser.add_argument('--perfect_attack', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--target_policy_path', type=str, default=None)  # log_2/PongNoFrameskip-v4/dqn/policy.pth
    parser.add_argument('--target_policy', type=str, default=None)  # dqn, a2c, ppo
    parser.add_argument('--min', type=float, default=0.)
    parser.add_argument('--max', type=float, default=1.)
    parser.add_argument('--steps', type=int, default=21)
    args = parser.parse_known_args()[0]
    return args


# python atari_uniform_attack_benchmark_radqn.py --task "PongNoFrameskip-v4" --resume_path "log/PongNoFrameskip-v4/radqn/policy.pth" --policy "dqn" --eps 0.1 --perfect_attack --test_num 1
# python atari_uniform_attack_benchmark_radqn.py --task "PongNoFrameskip-v4" --resume_path "log/PongNoFrameskip-v4/radqn/defended_policy.pth" --policy "dqn" --eps 0.1 --perfect_attack --test_num 1
def benchmark_adversarial_policy(args=get_args()):
    args.policy = 'radqn'
    env_config = 'defended_models/radial_dqn/config.json'
    setup_json = read_config(env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.task:
            env_conf = setup_json[i]
    args.env = args.task
    args.skip_rate = 4
    args.max_episode_length = 10000
    env = atari_env(args.task, env_conf, args)

    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space
    # should be N_FRAMES x H x W
    print("Observations shape: ", env.observation_space.shape)
    print("Actions shape: ", env.action_space)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # make policy
    policy = make_policy(args, args.policy, args.resume_path)
    # make target policy
    adv_net = make_victim_network(args, policy)
    # define observations adversarial attack
    obs_adv_atk, atk_type = make_img_adv_attack(args, adv_net, targeted=False)
    print("Attack type:", atk_type)

    policy = RadialDQNPolicyAdapter(policy, args.device)

    # define adversarial collector
    collector = uniform_attack_collector(policy, env, obs_adv_atk,
                                         perfect_attack=args.perfect_attack,
                                         device=args.device
                                         )
    atk_freq = np.linspace(args.min, args.max, args.steps, endpoint=True)
    n_attacks = []
    rewards = []
    for f in atk_freq:
        collector.atk_frequency = f
        test_adversarial_policy = collector.collect(n_episode=args.test_num)
        atk_freq_ = test_adversarial_policy['atk_rate(%)']
        rewards.append(test_adversarial_policy['rew'])
        n_attacks.append(test_adversarial_policy['n_atks'])
        print("attack frequency =", atk_freq_, "| n_attacks =", n_attacks[-1],
              "| n_succ_atks (%)", test_adversarial_policy['succ_atks(%)'],
              "| reward: ", rewards[-1])
        # pprint.pprint(test_adversarial_policy)
    log_path = os.path.join(args.logdir, args.task, args.policy,
                            "uniform_attack_" + atk_type + ".npy")

    # save results
    with open(log_path, 'wb') as f:
        np.save(f, atk_freq)
        np.save(f, n_attacks)
        np.save(f, rewards)
    print("Results saved to", log_path)


if __name__ == '__main__':
    benchmark_adversarial_policy(get_args())
