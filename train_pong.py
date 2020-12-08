import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from defended_models.radial_dqn import RadialDQNPolicy
from tianshou.utils.net.common import Net

task = 'CartPole-v0'
lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 8, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, collect_per_step = 1000, 10
writer = SummaryWriter('log/dqn')  # tensorboard is also supported!
device = "cuda:0"

# you can also try with SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

# you can define other net by following the API:
# https://tianshou.readthedocs.io/en/latest/tutorials/dqn.html#build-the-network
env = gym.make(task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(layer_num=2, state_shape=state_shape, action_shape=action_shape, device=device).to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

#policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
policy = RadialDQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq, k=0.5, eps=0.01, device=device)
train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(buffer_size))
test_collector = ts.data.Collector(policy, test_envs)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, collect_per_step,
    test_num, batch_size,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
print(f'Finished training! Use {result["duration"]}')

policy.eval()
policy.set_eps(eps_test)
collector = ts.data.Collector(policy, env)
res = collector.collect(n_episode=10, render=0.)
print(res)