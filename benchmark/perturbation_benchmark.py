import os
import torch
import argparse
import numpy as np
from drl_attacks.uniform_attack import uniform_attack_collector
from utils import make_policy, make_img_adv_attack, make_atari_env_watch, make_victim_network
from advertorch.attacks import *
import matplotlib.pyplot as plt


# python benchmark/perturbation_benchmark.py --task PongNoFrameskip-v4 --logdir log_benchmark --test-num 10
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
    parser.add_argument('--policy', type=str, default='dqn')  # dqn, a2c, ppo
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log_benchmark')
    parser.add_argument('--attack_freq', type=float, default=0.5)
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = get_args()
    args.resume_path = os.path.join("log", args.task, args.policy, "policy.pth")
    args.perfect_attack = False
    args.target_policy = args.policy
    env = make_atari_env_watch(args)
    img_attacks = [#"GradientAttack",  # too weak
                   "GradientSignAttack",  # ok
                   "LinfPGDAttack",  # ok
                   "MomentumIterativeAttack",  # ok
                    ]
    attack_labels = {"GradientAttack": "FGM",
                     "GradientSignAttack": "FGSM",
                     "LinfPGDAttack": "PGD-L1",
                     "L2PGDAttack": "PGD-L2",
                     "CarliniWagnerL2Attack": "CW",
                     "SparseL1DescentAttack": "SD",
                     "MomentumIterativeAttack": "MI",
                     "ElasticNetL1Attack": "EN",
                     }
    rl_defenses = ["",
                   "FGSMAdversarialTraining",
                   "PGDAdversarialTraining"]
    defense_labels = {"": "No Defense",
                      "FGSMAdversarialTraining": "FGSM AdvTr",
                      "PGDAdversarialTraining": "PGD AdvTr"}
    atk_eps = np.linspace(0., 0.05, 20, endpoint=True)
    save_path = os.path.join(args.logdir, args.task, args.policy)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f = open(os.path.join(args.logdir, args.task, "benchmark_result.txt"), "w+")

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape: ", args.state_shape)
    print("Actions shape: ", args.action_shape)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # make policy
    policy = make_policy(args, args.policy, args.resume_path)
    adv_net = make_victim_network(args, policy)
    # make defended policies
    for defense in rl_defenses:

        # chart
        fig, ax = plt.subplots()

        if defense == "":
            def_policy = policy
        else:
            def_policy = make_policy(args, args.policy,
                                     os.path.join("log_def", args.task, args.policy, defense + ".pth"))
        # use "def_policy" to make predictions and "adv_net" to craft adversarial attacks

        # define adversarial collector
        collector = uniform_attack_collector(def_policy, env,
                                             obs_adv_atk=None,
                                             perfect_attack=False,
                                             atk_frequency=args.attack_freq,
                                             device=args.device)
        for img_atk in img_attacks:

            rewards = []

            for eps in atk_eps:
                # define observations adversarial attack
                args.eps, args.image_attack = eps, img_atk
                collector.obs_adv_atk, _ = make_img_adv_attack(args, adv_net, targeted=False)
                test_adversarial_policy = collector.collect(n_episode=args.test_num)
                rewards.append(test_adversarial_policy['rew'])
            str_rewards = [str(x) for x in rewards]
            print(attack_labels[img_atk] + "-" + defense_labels[defense] + "-" + " ".join(str_rewards))
            f.write(attack_labels[img_atk] + "-" + defense_labels[defense] + "-" + " ".join(str_rewards) + "\n")

            plt.plot(atk_eps, rewards, label=attack_labels[img_atk])
        plt.xlabel('Perturbation Budget')
        plt.ylabel('Reward')
        plt.title(defense_labels[defense])
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(save_path, defense_labels[defense] + ".jpg"))
    f.close()
