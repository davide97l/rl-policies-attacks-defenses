from advertorch.attacks.base import Attack
import gym
import torch
from typing import Optional
from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_numpy
import numpy as np
import time
import warnings
from typing import Dict, List
import copy


class base_attack_collector:
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param obs_adv_atk: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack.
    :param perfect_attack: force adversarial attacks on observations to be
        always effective (ignore the ``adv`` param).
    """
    def __init__(self,
                 policy: BasePolicy,
                 env: gym.Env,
                 obs_adv_atk: Attack,
                 perfect_attack: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        self.device = device
        self.policy = policy
        self.env = env
        self.obs_adv_atk = obs_adv_atk
        self.perfect_attack = perfect_attack
        self.action_space = self.env.action_space.shape or self.env.action_space.n
        self.data = Batch(state={}, obs={}, act={}, rew={}, done={}, info={},
                          obs_next={}, policy={})
        self.reset_env()
        self.episode_count = 0  # current number of episodes
        self.reward_total = 0.  # total episode cumulative reward
        self.frames_count = 0  # number of observed frames
        self.n_attacks = 0  # number of attacks performed
        self.succ_attacks = 0  # number of successful image attacks
        self.start_time = 0  # time when the attack starts

    def reset_env(self):
        self.data.obs = self.env.reset()

    def render(self, **kwargs) -> None:
        return self.env.render(**kwargs)

    def reset_attack(self):
        self.episode_count, self.reward_total, self.frames_count,\
            self.n_attacks, self.succ_attacks = 0, 0, 0, 0, 0
        self.start_time = time.time()

    def get_attack_stats(self) -> Dict[str, float]:
        duration = max(time.time() - self.start_time, 1e-9)
        if self.episode_count == 0:
            self.episode_count = 1
        return {
            'n/ep': self.episode_count,
            'n/st': self.frames_count,
            'v/st': self.frames_count / duration,
            'v/ep': self.episode_count / duration,
            'rew': self.reward_total / self.episode_count,
            'len': self.frames_count / self.episode_count,
            'n_atks': self.n_attacks / self.episode_count,
            'n_succ_atks': self.succ_attacks / self.episode_count,
            'atk_rate(%)': self.n_attacks / self.frames_count,
            'succ_atks(%)': self.succ_attacks / self.n_attacks if self.n_attacks > 0 else 0,
        }

    def show_warning(self):
        if self.frames_count >= 100000 and self.episode_count == 0:
            warnings.warn(
                'There are already many steps in an episode. '
                'You should add a time limitation to your environment!',
                Warning)

    def check_end_attack(self, n_step, n_episode) -> bool:
        """Returns True when the attack terminates"""
        if n_step:
            if self.frames_count >= n_step:
                return True
        if n_episode:
            if self.episode_count >= n_episode:
                return True
        return False

    def perform_step(self):
        """
        Performs action 'self.data.act' on 'self.env' and store the next observation in 'self.data.obs'
        """
        obs_next, rew, done, info = self.env.step(self.data.act[0])
        self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)
        self.reward_total += rew
        if self.data.done:
            self.episode_count += 1
            self.reset_env()
        self.data.obs = self.data.obs_next

    def predict_next_action(self):
        """
        Predicts the next action given observation 'self.data.obs' and policy 'self.policy',
        and stores it in 'self.data.act'
        :return: outcome of policy forward pass
        """
        with torch.no_grad():
            self.data.obs = np.expand_dims(self.data.obs, axis=0)
            result = self.policy(self.data, last_state=None)
        self.data.act = to_numpy(result.act)
        return result

    def obs_attacks(self,
                    target_action: Optional[List[int]] = None,
                    ):
        """
        Performs an image adversarial attack on the observation stored in 'self.data.obs' respect to
        the action 'target_action' using the method defined in 'self.obs_adv_atk'
        :param target_action:
                - if obs_adv_atk.targeted=False, then 'target_action' must be the normal action.
                - if obs_adv_atk.targeted=True, then 'target_action' must be the adversarial action.
        """
        if not target_action:
            target_action = self.data.act
        obs = torch.FloatTensor(self.data.obs).to(self.device)  # convert observation to tensor
        act = torch.tensor(target_action).to(self.device)  # convert action to tensor
        adv_obs = self.obs_adv_atk.perturb(obs, act)  # create adversarial observation
        with torch.no_grad():
            data = copy.deepcopy(self.data)
            data.obs = adv_obs.cpu().detach().numpy()
            result = self.policy(data, last_state=None)
        self.data.act = to_numpy(result.act)

    def collect(self,
                n_step: int = 0,
                n_episode: int = 0,
                render: Optional[float] = None
                ) -> Dict[str, float]:
        """
        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect.
        :param float render: the sleep time between rendering consecutive
            frames, defaults to ``None`` (no rendering).
        :return: A dict including the following keys
            * ``n/ep`` the collected number of episodes.
            * ``n/st`` the collected number of steps.
            * ``v/st`` the speed of steps per second.
            * ``v/ep`` the speed of episode per second.
            * ``rew`` the mean reward over collected episodes.
            * ``len`` the mean length over collected episodes.
            * ``n_attacks`` number of performed attacks.
            * ``n_succ_attacks`` number of performed successful attacks.
            * ``n_attacks(%)`` ratio of performed attacks over steps.
            * ``succ_atks(%)`` ratio of successful attacks over performed attacks.
        """

        error = "Sub-classes must implement 'collect'."
        raise NotImplementedError(error)
