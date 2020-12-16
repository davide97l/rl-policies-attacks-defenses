import os
import time
from collections import deque

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from tianshou.data import Batch, to_numpy
from utils import make_policy, make_img_adv_attack, make_atari_env_watch, make_victim_network
import random as rd
from copy import deepcopy
from typing import Dict, List, Union, Optional, Callable


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

    assert args.resume_path is not None, \
        "You are training with adversarial training but you haven't declared a base trained model"

    if args.target_model_path:
        victim_policy = make_policy(args, args.algo, args.target_model_path)
    else:
        victim_policy = actor_critic

    args.target_policy, args.policy = args.algo, args.algo
    args.perfect_attack = False
    adv_net = make_victim_network(args, victim_policy)
    adv_atk, _ = make_img_adv_attack(args, adv_net, targeted=False)

    # watch agent's performance
    def watch():
        print("Testing agent ...")
        actor_critic.eval()
        args.task, args.frames_stack = args.env_name, 4
        env = make_atari_env_watch(args)
        obs = env.reset()
        n_ep, tot_rew = 0, 0
        succ_attacks, n_attacks = 0, 0
        while True:
            inputs = Batch(obs=np.expand_dims(obs, axis=0))
            with torch.no_grad():
                result = actor_critic(inputs)
            action = result.act

            # START ADVERSARIAL ATTACK
            x = rd.uniform(0, 1)
            if x < args.atk_freq:
                ori_act = action
                obs = torch.FloatTensor(inputs.obs).to(device)
                data = Batch(obs=obs)
                adv_act, adv_obs = obs_attacks(data, ori_act, adv_atk, actor_critic)
                for i in range(len(adv_act)):
                    if adv_act[i] != ori_act[i]:
                        succ_attacks += 1
                n_attacks += args.num_processes
                action = adv_act

            # Observe reward and next obs
            obs, reward, done, _ = env.step(action)
            tot_rew += reward
            if done:
                n_ep += 1
                obs = env.reset()
                if n_ep == args.test_num:
                    break
        if n_attacks == 0:
            n_attacks = 1
        print("Evaluation using {} episodes: mean reward {:.5f}, succ_atks(%) {:.3f}\n".format(
            n_ep, tot_rew / n_ep, succ_attacks / n_attacks))

    if args.watch:
        watch()
        exit(0)

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
    succ_attacks = 0
    n_attacks = 0
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        for step in range(args.num_steps):

            # Get action and value of original observation
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Given original action, generate adversarial observation
            x = rd.uniform(0, 1)
            if x < args.atk_freq:
                ori_act = action.flatten()
                data = Batch(obs=obs)
                adv_act, adv_obs = obs_attacks(data, ori_act, adv_atk, actor_critic)
                for i in range(len(adv_act)):
                    if adv_act[i] != ori_act[i]:
                        succ_attacks += 1
                n_attacks += len(adv_act)
                adv_obs = torch.FloatTensor(adv_obs).to(device)
                rollouts.obs[step] = adv_obs

            # Get action and value of adversarial observation
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs given adversarial action
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

            # Insert in memory adversarial action and value
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
            if n_attacks == 0:
                n_attacks = 1
            print(
                "Updates {}, num timesteps {}, FPS {} \nLast {} training episodes: mean/median reward {:.1f}/{:.1f},"
                " min/max reward {:.1f}/{:.1f} (best avg reward {:.1f}), succ_atks(%) {:.3f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), best_reward, succ_attacks / n_attacks))
            n_attacks, succ_attacks = 0, 0
    print("model saved to " + str(os.path.join(args.save_dir, args.algo, "policy.pth")))
    watch()


def obs_attacks(data: Batch,
                target_action: List[int],
                obs_adv_atk,
                policy
                ):
    """
    Performs an image adversarial attack on the observation stored in 'obs' respect to
    the action 'target_action' using the method defined in 'self.obs_adv_atk'
    """
    data = deepcopy(data)
    obs = data.obs
    act = target_action
    adv_obs = obs_adv_atk.perturb(obs, act)  # create adversarial observation
    with torch.no_grad():
        adv_obs = adv_obs.cpu().detach().numpy()
        data.obs = adv_obs
        result = policy(data, last_state=None)
    return to_numpy(result.act), adv_obs


if __name__ == "__main__":
    main()