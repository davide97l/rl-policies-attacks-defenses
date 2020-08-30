import matplotlib.pyplot as plt
import numpy as np

# Data
task = "Pong"
model = "dqn"
img_attack = "perfect_attack"
rl_attack = "uniform_attack"
input_file = ["log/" + task + "NoFrameskip-v4/" + model + "/" + rl_attack + "_" + img_attack + ".npy"]

atk_freq = []
n_attacks = []
rewards = []

i = 0
for file in input_file:
    with open(file, 'rb') as f:
        atk_freq.append(np.load(f))
        n_attacks.append(np.load(f))
        rewards.append(np.load(f))
    i += 1

print("Number attacks:", n_attacks[0])
print("Rewards:", rewards[0])

plt.plot(n_attacks[0], rewards[0])

# Add titles
plt.title(task + " - " + model + " - " + rl_attack, loc='center')
plt.xlabel("Number of attacks")
plt.ylabel("Final reward")

plt.show()