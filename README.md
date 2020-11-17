# Tianshou Reinforcement Learning Adversarial Attacks
This repository implements some classic adversarial attack methods for deep reinforcement learning agents including:
- Uniform attack [[link](https://arxiv.org/abs/1702.02284)].
- Strategical timed attack [[link](https://www.ijcai.org/Proceedings/2017/0525.pdf)].
- Critical point attack [[link](https://arxiv.org/abs/2005.07099)].
- Critical strategy attack.
- Adversarial policy attack [[link](https://arxiv.org/abs/1905.10615)].

## Available models
It also makes available trained models for different tasks which can be found in the folder `log`. The following table reports their average score, the commands used to train them and their corresponding algorithm.

| task                        | DQN   | A2C   | PPO   |
|-----------------------------|-------|-------|-------|
| PongNoFrameskip-v4          | 20    | 20    | 21    |
| BreakoutNoFrameskip-v4      | 349   | 400   | 470   |
| EnduroNoFrameskip-v4        | 751   | NA    | 1064  |
| QbertNoFrameskip-v4         | 4382  | 7762  | 14580 | 
| MsPacmanNoFrameskip-v4      | 2787  | 2230  | 1929  |
| SpaceInvadersNoFrameskip-v4 | 640   | 856   | 1120  |
| SeaquestNoFrameskip-v4      | NA    | 1610  | 1798  |

## Transferability over policies
This section shows the performance of different adversarial attacks methods and their comparison between attacking an agent and a surrogate agent trained with the same policy (transfer policy).

![](results/pong_uniform_attack_dqn_fgm_eps_01_transfer_dqn.png)
![](results/pong_strategically_timed_attack_dqn_fgm_eps_01_transfer_dqn.png)
![](results/pong_critical_strategy_attack_dqn_fgm_eps_01_transfer_dqn.png)
![](results/pong_critical_point_attack_dqn_fgm_eps_01_transfer_dqn.png)
![](results/pong_adversarial_policy_attack_dqn_fgm_eps_01_transfer_dqn.png)
