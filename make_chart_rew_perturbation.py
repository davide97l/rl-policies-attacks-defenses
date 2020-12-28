import matplotlib.pyplot as plt
import numpy as np
import os
from statistics import mean


def smooth(list1, list2, smoothing=4):
    """Smooth list2 according to list1"""
    poly = np.polyfit(list1, list2, smoothing)
    list_2 = np.poly1d(poly)(list1)
    return list_2


if __name__ == '__main__':
    benchmark_path = "log_perturbation_benchmark\PongNoFrameskip-v4\dqn\perturbation_benchmark_result.txt"
    atk_eps = np.linspace(0., 0.05, 10, endpoint=True)
    save_path = os.path.join("log_perturbation_benchmark\PongNoFrameskip-v4\dqn")
    smoothing = 3

    f = open(benchmark_path, 'r')
    attacks, defenses, rewards = [], [], []
    for line in f.readlines():
        try:
            attack, defense, reward = line.split("|")[0], line.split("|")[1], line.split("|")[2].split()
            reward = [float(rew) for rew in reward]
        except Exception:
            pass
        attacks.append(attack)
        defenses.append(defense)
        rewards.append(reward)
    print(defenses)

    fig, ax = plt.subplots()
    for i, defense in enumerate(defenses):
        if attacks[i] == "No Attack":
            rewards[i] = [mean(rewards[i]) for _ in rewards[i]]
        #else:
            #rewards[i] = smooth(atk_eps, rewards[i], smoothing=smoothing)
        plt.plot(atk_eps, rewards[i], label=attacks[i])
        if i == len(defenses) - 1 or defenses[i] != defenses[i+1]:
            plt.xlabel('Perturbation Budget')
            plt.ylabel('Reward')
            plt.title(defense)
            plt.legend(loc='upper right')
            plot_path = os.path.join(save_path, defense.replace(" ", "_") + ".jpg")
            plt.savefig(plot_path)
            print("plotted:", plot_path)
            #plt.show()
            last_defense = defense
            fig, ax = plt.subplots()
    f.close()

