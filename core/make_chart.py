import matplotlib.pyplot as plt
import numpy as np

# Data
input_file = ["log/PongNoFrameskip-v4/dqn/uniform_attack_perfect_attack.npy",
              "log/PongNoFrameskip-v4/dqn/uniform_attack_fgsm_eps_0.1.npy",
              "log/PongNoFrameskip-v4/dqn/uniform_attack_fgsm_eps_0.2.npy",
              "log/PongNoFrameskip-v4/dqn/uniform_attack_fgsm_eps_0.3.npy"]

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

plt.plot(atk_freq[0], rewards[0], label='perfect attack')
plt.plot(atk_freq[0], rewards[1], label='eps=0.1')
plt.plot(atk_freq[0], rewards[2], label='eps=0.2')
plt.plot(atk_freq[0], rewards[3], label='eps=0.3')

# Add legend
plt.legend()

# Add titles
plt.title("Pong - DQN", loc='center')
plt.xlabel("Attack frequency")
plt.ylabel("Final reward")

plt.show()

