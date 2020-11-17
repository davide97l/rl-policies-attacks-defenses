# Tianshou Reinforcement Learning Adversarial Attacks
This repository implements some classic adversarial attack methods for deep reinforcement learning agents including:
- Uniform attack [[link](https://arxiv.org/abs/1702.02284)].
- Strategical timed attack [[link](https://www.ijcai.org/Proceedings/2017/0525.pdf)].
- Critical point attack [[link](https://arxiv.org/abs/2005.07099)].
- Critical strategy attack.
- Adversarial policy attack [[link](https://arxiv.org/abs/1905.10615)].

## Available models
It also makes available trained models for different tasks which can be found in the folder `log`. The following table reports their average score for three different algorithms: DQN, A2C and PPO.

| task                        | DQN   | A2C   | PPO   |
|-----------------------------|-------|-------|-------|
| PongNoFrameskip-v4          | 20    | 20    | 21    |
| BreakoutNoFrameskip-v4      | 349   | 400   | 470   |
| EnduroNoFrameskip-v4        | 751   | NA    | 1064  |
| QbertNoFrameskip-v4         | 4382  | 7762  | 14580 | 
| MsPacmanNoFrameskip-v4      | 2787  | 2230  | 1929  |
| SpaceInvadersNoFrameskip-v4 | 640   | 856   | 1120  |
| SeaquestNoFrameskip-v4      | NA    | 1610  | 1798  |

## Usage
**Train** DQN agent to play Pong.
```
  python atari_dqn.py --task "PongNoFrameskip-v4"
```
**Train** A2C agent to play Breakout.
```
  python atari_a2c_ppo.py --env-name "BreakoutNoFrameskip-v4" --algo a2c
```
**Train** PPO agent to play Breakout.
```
  python atari_a2c_ppo.py --env-name "BreakoutNoFrameskip-v4"--algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
```
**Test** DQN agent playing Pong.
```
  python atari_dqn.py --resume_path "log/PongNoFrameskip-v4/dqn/policy.pth" --watch --test_num 10 --task "PongNoFrameskip-v4"
```
**Test** A2C agent playing Breakout.
```
  python atari_a2c_ppo.py --env-name "BreakoutNoFrameskip-v4" --algo a2c --resume_path "log/BreakoutNoFrameskip-v4/a2c/policy.pth" --watch --test_num 10
```
**Test** PPO agent playing Breakout.
```
  python atari_a2c_ppo.py --env-name "BreakoutNoFrameskip-v4" --algo ppo --resume_path "log/BreakoutNoFrameskip-v4/ppo/policy.pth" --watch --test_num 10
```
**Train** DQN malicious agent to play Pong minimizing the score.
```
  python atari_dqn.py --task "PongNoFrameskip-v4" --invert_reward --epoch 1
```
**Attack** Pong-DQN with Uniform Attack and FGSM with `eps=0.1`.
```
  python atari_uniform_attack_benchmark.py --task "PongNoFrameskip-v4" --resume_path "log/PongNoFrameskip-v4/dqn/policy.pth" --policy "dqn" --eps 0.1
```
**Attack** Breakout-A2C with Adversarial Policy Attack and FGSM according to a DQN-adversarial policy and performing attacks on a DQN surrogate model.
```
  python atari_adversarial_policy_attack_benchmark.py --task "BreakoutNoFrameskip-v4" --resume_path "log/BreakoutNoFrameskip-v4/a2c/policy.pth" --adv_policy "dqn" --adv_policy_path "log_adv_policy/BreakoutNoFrameskip-v4/dqn/policy.pth" --policy "a2c" --target_policy_path "log_2/BreakoutNoFrameskip-v4/a2c/policy.pth" --target_policy "a2c" --eps 0.01 --min 0.95 --max 1. --steps 11 --test_num 10
```
Moreover, you should see files `atari_attack_name_benchmark.py` to understand how to perform a specific attack.


## Test transferability over policies
This section shows the performance of different adversarial attacks methods and their comparison between attacking an agent and 2 surrogate agents: one trained with the same policy and one trained on a different algorithm.

![](results/dqn_transfer_dqn_a2c/pong_uniform_attack_dqn_fgm_eps_01_transfer_dqn_a2c.png)
![](results/dqn_transfer_dqn_a2c/pong_strategically_timed_attack_dqn_fgm_eps_01_transfer_dqn_a2c.png)
![](results/dqn_transfer_dqn_a2c/pong_critical_strategy_attack_dqn_fgm_eps_01_transfer_dqn_a2c.png)
![](results/dqn_transfer_dqn_a2c/pong_critical_point_attack_dqn_fgm_eps_01_transfer_dqn_a2c.png)
![](results/dqn_transfer_dqn_a2c/pong_adversarial_policy_attack_dqn_fgm_eps_01_transfer_dqn_a2c.png)
