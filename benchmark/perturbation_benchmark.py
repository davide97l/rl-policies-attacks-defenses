import os
import torch
import argparse
import numpy as np
from drl_attacks import *
from utils import make_policy, make_img_adv_attack, make_atari_env_watch, make_victim_network
from advertorch.attacks import *
from img_defenses import *


# python perturbation_benchmark.py --task PongNoFrameskip-v4 --logdir log_benchmark --test-num 5
# python perturbation_benchmark.py --targeted --device "cuda:1" --test-num 5
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
    parser.add_argument('--logdir', type=str, default='log_perturbation_benchmark')
    parser.add_argument('--attack_freq', type=float, default=0.5)
    parser.add_argument('--sample_points', type=int, default=10)
    parser.add_argument('--targeted', default=False, action='store_true')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = get_args()
    args.resume_path = os.path.join("log", args.task, args.policy, "policy.pth")
    args.perfect_attack = False
    args.target_policy = args.policy
    env = make_atari_env_watch(args)
    # comment the attacks you don't need
    img_attacks = [#"No Attack",
                   "GradientSignAttack",  # ok
                   #"LinfPGDAttack",  # ok
                   "MomentumIterativeAttack",  # ok
                   #"DeepfoolLinfAttack"
                  ]
    # you can change attack parameters in the utils.py file

    attack_labels = {"No Attack": "No Attack",
                     "GradientAttack": "FGM",
                     "GradientSignAttack": "FGSM-Linf",
                     "LinfPGDAttack": "PGD-Linf",
                     "L2PGDAttack": "PGD-L2",
                     "CarliniWagnerL2Attack": "CW",
                     "SparseL1DescentAttack": "SD",
                     "MomentumIterativeAttack": "MI-Linf",
                     "ElasticNetL1Attack": "EN",
                     "DeepfoolLinfAttack": "Deepfool-Linf"
                     }
    # comment the defences you don't need
    rl_defenses = [#"No Defense",
                   #"FGSMAdversarialTraining",
                   "PGDAdversarialTraining",
                   #"JPEGFilter",
                   #"BitSqueezing",
                   #"Smoothing",
                   ]
    defense_labels = {"No Defense": "No Defense",
                      "FGSMAdversarialTraining": "FGSM AdvTr",
                      "PGDAdversarialTraining": "PGD AdvTr",
                      "JPEGFilter": "JPEG Filter",
                      "BitSqueezing": "Bit Squeezing",
                      "Smoothing": "Smoothing"}
    atk_eps = np.linspace(0., 0.05, args.sample_points, endpoint=True)
    if args.targeted is False:
        save_path = os.path.join(args.logdir, args.task, args.policy, "untargeted")
    else:
        save_path = os.path.join(args.logdir, args.task, args.policy, "targeted")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = "perturbation_benchmark_result.txt"
    if len(rl_defenses) == 1:
        file_name = "perturbation_benchmark_" + str(rl_defenses[0]) + ".txt"
    f_rew = open(os.path.join(save_path, file_name), "w+")

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

        if defense == "No Defense":
            def_policy = policy
        elif "AdversarialTraining" in defense:
            def_policy = make_policy(args, args.policy,
                                     os.path.join("log_def", args.task, args.policy, defense + ".pth"))
        elif "JPEGFilter" == defense:
            def_policy = JPEGFilterDefense(policy, quality=20)
        elif "BitSqueezing" == defense:
            def_policy = BitSqueezingDefense(policy, bit_depth=5)
        elif "Smoothing" == defense:
            def_policy = SmoothingDefense(policy, kernel_size=2, smoothing="median")
        else:
            raise Exception("Defense not defined")
        # use "def_policy" to make predictions and "adv_net" to craft adversarial attacks

        # define adversarial collector
        if args.targeted is False:
            collector = uniform_attack_collector(def_policy, env,
                                                 obs_adv_atk=None,
                                                 perfect_attack=False,
                                                 atk_frequency=args.attack_freq,
                                                 device=args.device)
        else:
            collector = strategically_timed_attack_collector(def_policy, env,
                                                             obs_adv_atk=None,
                                                             perfect_attack=False,
                                                             softmax=False,
                                                             beta=0,  # in this way it always attacks, ignore args.frequency
                                                             device=args.device)
        for img_atk in img_attacks:

            rewards = []
            accuracies = []

            collector.atk_frequency = args.attack_freq

            for eps in atk_eps:
                # define observations adversarial attack
                args.eps, args.image_attack = eps, img_atk
                episodes = args.test_num
                if img_atk == "No Attack":
                    # we can assign a random attack since the frequency is 0
                    args.image_attack = "GradientSignAttack"
                    collector.atk_frequency = 0
                    episodes *= 2

                if collector.atk_frequency > 0 or len(rewards) == 0:
                    collector.obs_adv_atk, _ = make_img_adv_attack(args, adv_net, targeted=args.targeted)
                    test_adversarial_policy = collector.collect(n_episode=episodes)
                    rewards.append(test_adversarial_policy['rew'])
                    accuracies.append(test_adversarial_policy['succ_atks(%)'])
                else:
                    rewards.append(rewards[-1])
                    accuracies.append(accuracies[-1])

            str_rewards = [str(x) for x in rewards]
            str_accuracies = [str(x) for x in accuracies]
            print(attack_labels[img_atk] + "|" + defense_labels[defense]
                  + "|" + " ".join(str_rewards) + "|" + " ".join(str_accuracies))
            f_rew.write(attack_labels[img_atk] + "|" + defense_labels[defense]
                        + "|" + " ".join(str_rewards) + "|" + " ".join(str_accuracies) + "\n")
    f_rew.close()
