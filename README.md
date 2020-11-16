# Tianshou Reinforcement Learning Adversarial Attacks
This repository implements some classic adversarial attack methods for deep reinforcement learning agents including:
- Uniform attack [[link](https://arxiv.org/abs/1702.02284)].
- Strategical timed attack [[link](https://www.ijcai.org/Proceedings/2017/0525.pdf)].
- Critical point attack [[link](https://arxiv.org/abs/2005.07099)].
- Critical strategy attack.
- Adversarial policy attack [[link](https://arxiv.org/abs/1905.10615)].

## Available models
It also makes available trained models for different tasks which can be found in the folder `log`. The following table reports their average score, the commands used to train them and their corresponding algorithm.

| task                        | best reward | parameters                                                   | algorithm           |
| --------------------------- | ----------- | ------------------------------------------------------------ | ------------------- |
| PongNoFrameskip-v4          | 20          | `python3 atari_dqn.py --task "PongNoFrameskip-v4" --test_num 10` | DQN |
| BreakoutNoFrameskip-v4      | 349         | `python3 atari_dqn.py --task "BreakoutNoFrameskip-v4" --test_num 10`  | DQN    |
| EnduroNoFrameskip-v4        | 751         | `python3 atari_dqn.py --task "EnduroNoFrameskip-v4" --test_num 10`  | DQN    |
| QbertNoFrameskip-v4         | 4382        | `python3 atari_dqn.py --task "QbertNoFrameskip-v4" --test_num 10`  | DQN    |
| MsPacmanNoFrameskip-v4      | 2787        | `python3 atari_dqn.py --task "MsPacmanNoFrameskip-v4" --test_num 10`  | DQN    |
| SpaceInvadersNoFrameskip-v4 | 640         | `python3 atari_dqn.py --task "SpaceInvadersNoFrameskip-v4" --test_num 10`  | DQN    |
| SeaquestNoFrameskip-v4      | NA          | `python3 atari_dqn.py --task "SeaquestNoFrameskip-v4" --test_num 10`  | DQN    |
| PongNoFrameskip-v4          | 20          | `python atari_a2c_ppo.py --env-name "PongNoFrameskip-v4" --algo a2c` | A2C |
| BreakoutNoFrameskip-v4      | 400         | `python atari_a2c_ppo.py --env-name "BreakoutNoFrameskip-v4" --algo a2c`  | A2C    |
| EnduroNoFrameskip-v4        | NA          | `python atari_a2c_ppo.py --env-name "EnduroNoFrameskip-v4" --algo a2c`  | A2C    |
| QbertNoFrameskip-v4         | 7762        | `python atari_a2c_ppo.py --env-name "QbertNoFrameskip-v4" --algo a2c`  | A2C    |
| MsPacmanNoFrameskip-v4      | 2230        | `python atari_a2c_ppo.py --env-name "MsPacmanNoFrameskip-v4" --algo a2c`  | A2C    |
| SpaceInvadersNoFrameskip-v4 | 856         | `python atari_a2c_ppo.py --env-name "SpaceInvaderNoFrameskip-v4" --algo a2c`  | A2C    |
| SeaquestNoFrameskip-v4      | 1610        | `python atari_a2c_ppo.py --env-name "SeaquestNoFrameskip-v4" --algo a2c`  | A2C    |

## Transferability over policies
This section shows the performance of different adversarial attacks methods and their comparison between attacking an agent and a surrogate agent trained with the same policy (transfer policy).

![](results/pong_uniform_attack_dqn_fgm_eps_01_transfer_dqn.png)
![](results/pong_strategically_timed_attack_dqn_fgm_eps_01_transfer_dqn.png)
![](results/pong_critical_strategy_attack_dqn_fgm_eps_01_transfer_dqn.png)
![](results/pong_critical_point_attack_dqn_fgm_eps_01_transfer_dqn.png)
![](results/pong_adversarial_policy_attack_dqn_fgm_eps_01_transfer_dqn.png)
