import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from utils import make_atari_env_watch
from tianshou.data import Batch
from utils import make_policy


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = args.device

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, None, device, False)

    if args.resume_path is None:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            device=args.device,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)
        actor_critic.init(device)
    else:
        actor_critic = make_policy(args, args.algo, args.resume_path)

    # watch agent's performance
    def watch():
        print("Testing agent ...")
        actor_critic.eval()
        args.task, args.frames_stack = args.env_name, 4
        env = make_atari_env_watch(args)
        obs = env.reset()
        n_ep, tot_rew = 0, 0
        while True:
            inputs = Batch(obs=np.expand_dims(obs, axis=0))
            with torch.no_grad():
                result = actor_critic(inputs)
            action = result.act
            # Observe reward and next obs
            obs, reward, done, _ = env.step(action)
            tot_rew += reward
            if done:
                n_ep += 1
                obs = env.reset()
                if n_ep == args.test_num:
                    break
        print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
            n_ep, tot_rew / n_ep))

    if args.watch:
        watch()
        exit(0)

    if args.resume_path is not None:
        args.rms_eps = 0.1

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.rms_eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.rms_eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    acc_rewards = np.zeros(args.num_processes)
    best_reward = -np.inf

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    print("start training")
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i, d in enumerate(done):
                acc_rewards[i] += reward[i].detach().cpu()[0]
                if d:
                    episode_rewards.append(acc_rewards[i])
                    acc_rewards[i] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if len(episode_rewards) > 0 and np.mean(episode_rewards) >= best_reward and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            best_reward = np.mean(episode_rewards)
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, "policy.pth"))

        if j % args.log_interval == 0 and len(episode_rewards) > 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \nLast {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} (best avg reward {:.1f})\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), best_reward))
    print("model saved to " + str(os.path.join(args.save_dir, args.algo, "policy.pth")))
    watch()


if __name__ == "__main__":
    main()