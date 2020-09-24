import matplotlib.pyplot as plt
import numpy as np


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


def limit_lists(list1, x_lists, y_lists):
    """All the lists in x_lists will be in the same range of list1"""
    limit = list1[-1]
    n_lists = len(x_lists)
    for i in range(n_lists):
        x_lists[i] = ([l for l in x_lists[i] if l <= limit])
        y_lists[i] = y_lists[i][:len(x_lists[i])]
    return x_lists, y_lists


if __name__ == '__main__':

    # Data
    task = "Pong"
    model = "dqn"
    transfer_model = "dqn"
    img_attack = "fgm_eps_0.1"  # fgm_eps_0.05, perfect_attack, fgm_eps_0.3
    rl_attack = "critical_strategy_attack"  # strategically_timed_attack, uniform_attack, critical_strategy_attack
    has_atk_freq = 1
    has_n_attacks = 0
    smoothing = 4
    input_file = [
        "log/" + task + "NoFrameskip-v4/" + model + "/" + rl_attack + "_" + img_attack + ".npy",
        "log/" + task + "NoFrameskip-v4/" + model + "/" + rl_attack + "_" + img_attack + "_transf_" + transfer_model + ".npy",
        "log/" + task + "NoFrameskip-v4/" + model + "/" + rl_attack + "_" + "perfect_attack" + ".npy",
                  ]

    n_lines = len(input_file)
    atk_freq = []
    n_attacks = []
    rewards = []
    for file in input_file:
        with open(file, 'rb') as f:
            atk_freq.append(np.load(f))
            n_attacks.append(np.load(f))
            rewards.append(np.load(f))

    assert has_atk_freq or has_n_attacks
    if has_atk_freq:
        x = atk_freq
    elif has_n_attacks:
        x = n_attacks

    x1_lists = []
    y1_lists = []
    for i in range(n_lines):
        x[i], rewards[i] = sort_pivot(x[i], rewards[i])
        #print("Attack frequencies:", atk_freq[i])
        #print("Rewards:", rewards[i])
        #print("Number attacks:", n_attacks[i])
        rewards[i] = smooth(x[i], rewards[i], smoothing=smoothing)
        if i > 0:
            x1_lists.append(x[i])
            y1_lists.append(rewards[i])

    x_lists, y_lists = limit_lists(x[0], x1_lists, y1_lists)
    for i in range(1, n_lines):
        x[i] = x_lists[i-1]
        rewards[i] = y_lists[i-1]

    plt.plot(x[0], rewards[0], label="Policy")
    plt.plot(x[1], rewards[1], label="Transfer Policy")
    plt.plot(x[2], rewards[2], label="Perfect Attack")

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

