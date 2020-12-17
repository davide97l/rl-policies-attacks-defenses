## Usage
Before start using this repository, install the required libraries in the `requirements.txt` file.
```
  pip install -r requirements.txt"
```
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
**Defend** Pong DQN agent with **adversarial training**.
```
 python atari_adversarial_training_dqn.py --task "PongNoFrameskip-v4" --resume_path "log/PongNoFrameskip-v4/dqn/policy.pth" --logdir log_def --eps 0.01 --image_attack fgm
```
**Test** defended Pong DQN agent.
```
python atari_adversarial_training_dqn.py --task "PongNoFrameskip-v4" --resume_path "log_def/PongNoFrameskip-v4/dqn/policy.pth" --eps 0.01 --image_attack fgm --target_model_path log/PongNoFrameskip-v4/dqn/policy.pth --watch --test_num 10
```
**Defend** Pong A2C agent with **adversarial training**.
```
python atari_adversarial_training_a2c_ppo.py --env-name "PongNoFrameskip-v4" --num-env-steps 10000000 --algo a2c --resume_path "log/PongNoFrameskip-v4/a2c/policy.pth" --save-dir log_def --eps 0.01 --image_attack fgm --num-processes 16 --test_num 10
```
**Test** defended Pong A2C agent.
```
python atari_adversarial_training_a2c_ppo.py --env-name "PongNoFrameskip-v4" --algo a2c --resume_path "log_def/PongNoFrameskip-v4/a2c/policy.pth" --eps 0.01 --image_attack fgm --target_model_path "log/PongNoFrameskip-v4/a2c/policy.pth" --watch --test_num 10
```
**Defend** Pong PPO agent with **adversarial training**.
```
python atari_adversarial_training_a2c_ppo.py --env-name "PongNoFrameskip-v4" --num-env-steps 10000000 --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --resume_path "log/PongNoFrameskip-v4/ppo/policy.pth" --save-dir log_def --eps 0.01 --image_attack fgm --num-processes 8 --atk_freq 0.5 --test_num 10
```
**Test** defended Pong PPO agent.
```
python atari_adversarial_training_a2c_ppo.py --env-name "PongNoFrameskip-v4" --algo ppo --resume_path "log_def/PongNoFrameskip-v4/ppo/policy.pth" --eps 0.01 --image_attack fgm --target_model_path "log/PongNoFrameskip-v4/ppo/policy.pth" --watch --test_num 10
```
To understand how to perform adversarial attacks refer to the `example.ipynb` file and to the benchmark examples contained in the folder `benchmark`.
