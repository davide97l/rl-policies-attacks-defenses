import matplotlib.pyplot as plt
import numpy as np
import sys


def sort_pivot(list1, list2):
    """Sort list1, then sort list 2 according to the permutation of list1"""
    ind = np.argsort(list1)
    list1 = np.take_along_axis(list1, ind, axis=0)
    list2 = np.take_along_axis(list2, ind, axis=0)
    return list1, list2


def smooth(list1, list2, smoothing=4):
    """Smooth list2"""
    poly = np.polyfit(list1, list2, smoothing)
    list_2 = np.poly1d(poly)(list1)
    return list_2


def limit_lists(limit_freq, x_lists, y_lists):
    """All the lists in x_lists will be in the same range of list1"""
    n_lists = len(x_lists)
    for i in range(n_lists):
        x_lists[i] = ([l for l in x_lists[i] if l <= limit_freq])
        temp = y_lists[i][-1]
        y_lists[i] = y_lists[i][:len(x_lists[i])]
        y_lists[i] = np.concatenate([y_lists[i], [temp]])
        x_lists[i] = np.concatenate([x_lists[i], [limit_freq]])
    return x_lists, y_lists


if __name__ == '__main__':

    # Data
    task = "Pong"  # Pong
    model = "ppo"
    n_lines = 4
    transfer_model = "ppo"
    transfer_model_2 = "ppo"
    transfer_model_3 = "dqn"
    transfer_model_4 = "a2c"
    img_attack = "fgm_eps_0.05"  # fgm_eps_0.05, perfect_attack, fgm_eps_0.3
    rl_attack = "uniform_attack"  # strategically_timed_attack, uniform_attack, critical_strategy_attack, critical_point_attack, adversarial_policy_attack
    has_atk_freq = 1
    has_n_attacks = 0
    smoothing = 4
    limit_freq = 0.99
    min_freq = 0.0
    first_equal = False  # make first reward same as first line for all lines
    logdir = 'log_def'
    input_file = [
        logdir + "/" + task + "NoFrameskip-v4/" + model + "/" + rl_attack + "_" + img_attack + "_transf_" + transfer_model + ".npy",
        logdir + "/" + task + "NoFrameskip-v4/" + model + "/" + rl_attack + "_" + img_attack + "_transf_" + transfer_model_2 + "_2.npy",
        logdir + "/" + task + "NoFrameskip-v4/" + model + "/" + rl_attack + "_" + img_attack + "_transf_" + transfer_model_3 + ".npy",
        logdir + "/" + task + "NoFrameskip-v4/" + model + "/" + rl_attack + "_" + img_attack + "_transf_" + transfer_model_4 + ".npy",
    ]

    if n_lines is None:
        n_lines = len(input_file)
    else:
        input_file = input_file[:n_lines]
    atk_freq = []
    n_attacks = []
    rewards = []
    for file in input_file:
        with open(file, 'rb') as f:
            atk_freq.append(np.load(f))
            n_attacks.append(np.load(f))
            rewards.append(np.load(f))

    assert has_atk_freq or has_n_attacks
    if has_atk_freq and has_n_attacks:
        sys.exit(0)
    if has_atk_freq:
        x = atk_freq
    elif has_n_attacks:
        x = n_attacks

    x1_lists = []
    y1_lists = []
    min_x, min_index = np.inf, 0
    for i in range(n_lines):
        x[i], rewards[i] = sort_pivot(x[i], rewards[i])
        rewards[i] = smooth(x[i], rewards[i], smoothing=smoothing)

        if limit_freq:
            a = [(j, rewards[i][h]) for h, j in enumerate(x[i]) if j <= limit_freq]
            rewards[i] = [e[1] for e in a]
            x[i] = [e[0] for e in a]
        if min_freq:
            a = [(j, rewards[i][h]) for h, j in enumerate(x[i]) if j >= min_freq]
            rewards[i] = [e[1] for e in a]
            x[i] = [e[0] for e in a]

        x1_lists.append(x[i])
        y1_lists.append(rewards[i])
        if i == n_lines-1 and x[i][-1] < min_x:
            min_x = x[i][-1]
            min_index = i
        if first_equal:
            rewards[i][0] = max(max(rewards[0]), max(rewards[1]), max(rewards[2]), max(rewards[3]))

    x_lists, y_lists = limit_lists(min_x, x1_lists, y1_lists)
    for i in range(0, n_lines):
        x[i] = x_lists[i]
        rewards[i] = y_lists[i]

    plt.plot(x[0], rewards[0], label="Policy (" + transfer_model + ")")
    if n_lines > 1:
        plt.plot(x[1], rewards[1], label="Transfer Policy (" + transfer_model_2 + ")")
    if n_lines > 2:
        plt.plot(x[2], rewards[2], label="Transfer Algorithm (" + transfer_model_3 + ")")
    if n_lines > 3:
        plt.plot(x[3], rewards[3], label="Transfer Algorithm (" + transfer_model_4 + ")")

    plt.legend(loc='upper right')

    # Add titles
    rl_attack = rl_attack.replace("_", " ")
    plt.title(task + " - " + model + " - " + rl_attack, loc='center')
    if has_atk_freq:
        plt.xlabel("Attack frequency")
    if has_n_attacks:
        plt.xlabel("Number of attacks")
    plt.ylabel("Final reward")

    plt.show()
