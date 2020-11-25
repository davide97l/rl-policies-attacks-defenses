from drl_attacks.critical_strategy_attack import critical_strategy_attack_collector
from drl_attacks.base_attack import base_attack_collector
from advertorch.attacks.base import Attack
import gym
import time
import torch
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer, to_numpy
import itertools
import copy


class critical_point_attack_collector(critical_strategy_attack_collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param obs_adv_atk: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack.
    :param perfect_attack: force adversarial attacks on observations to be
        always effective (ignore the ``adv`` param).
    :param n: int, number of attacks in a sequence.
    :param m: int, number of observations in a sequence.
    :param delta: float, minimum DAM margin to consider an adversarial
        sequence respect to the standard sequence.
    :param acts_mask: list(int), contains a subset of actions to use during the
        exploration of the adversarial policies
    :param repeat_adv_act: int, during the process of searching for some adversarial strategies,
        repeat each adversarial action ``repeat_adv_act`` times. This will reduce the number of total
        action combinations to try but could influence the performance of the attack. Setting this
        parameter != 1 will cause m = m * repeat_adv_act and n = n * repeat_adv_act.
    :param atari: bool, specify whether ``env`` is an Atari game (True) or not (False), this parameters is
        used by the functions store_env_state and load_env_state in order to assume different behaviors
    :param full_search: bool, if False the adversarial search terminates after the first adversarial policy is found
    :param dam: danger aware metrics. Higher DAM values correspond to worst states
    """

    def __init__(self,
                 policy: BasePolicy,
                 env: gym.Env,
                 obs_adv_atk: Attack,
                 perfect_attack: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 n: int = 5,
                 m: int = None,
                 delta: float = 0.,
                 acts_mask: List[int] = None,
                 repeat_adv_act: int = 1,
                 atari: bool = True,
                 full_search: bool = True,
                 dam: Callable = None
                 ):
        super().__init__(
            policy, env, obs_adv_atk, perfect_attack, device, n, m,
            delta, acts_mask, repeat_adv_act, atari, full_search)

        self.dam = dam

    def adversarial_policy(self, batch):
        """Find an adversarial policy (return the standard policy if can't find better policies).
        :return: list of ints, an array containing the adversarial actions"""
        batch = copy.deepcopy(batch)
        batch.obs = np.squeeze(batch.obs, axis=0)
        action_shape = self.env.action_space.shape or self.env.action_space.n
        action_shape = np.prod(action_shape)
        if self.acts_mask:
            actions = [a for a in range(int(action_shape)) if a in self.acts_mask]
        else:
            actions = [a for a in range(int(action_shape))]
        atk_strategies = [p for p in itertools.product(actions, repeat=self.n // self.repeat_adv_act)]  # define attack strategies
        atk_strategies = np.repeat(atk_strategies, self.repeat_adv_act, axis=-1)
        if isinstance(self.env, gym.wrappers.Monitor):
            env = copy.deepcopy(self.env.env)  # copy the environment
        else:
            env = copy.deepcopy(self.env)
        self.store_env_state(self.env)  # store the state of the environment
        env = self.load_env_state(env)  # restore the state of the environment
        adv_acts = []  # actions of the best adversarial policy
        # test standard policy
        attack = False
        for i in range(self.m):
            with torch.no_grad():
                batch.obs = np.expand_dims(batch.obs, axis=0)
                result = self.policy(batch, None)
            act = to_numpy(result.act)
            obs_next, rew, done, info = env.step(act[0])
            adv_acts.append(act[0])
            if done:
                break
            batch = Batch(state={}, obs=obs_next, act={}, rew={}, done={}, info={},
                          obs_next={}, policy={})
        std_dam = self.dam(batch.obs[-1]) if self.dam is not None else rew
        best_dam = std_dam
        # test adversarial policies
        for atk in atk_strategies:
            self.store_env_state(self.env)  # store the state of the environment
            env = self.load_env_state(env)  # restore the state of the environment
            acts = list(atk)
            for act in acts:  # play n steps according to adversarial policy
                obs_next, rew, done, info = env.step(act)
                if done:
                    break
                batch = Batch(state={}, obs=obs_next, act={}, rew={}, done={}, info={},
                              obs_next={}, policy={})
            if self.m > self.n and not done:  # play n-m steps according to standard policy
                for i in range(self.m - self.n):
                    with torch.no_grad():
                        batch.obs = np.expand_dims(batch.obs, axis=0)
                        result = self.policy(batch, None)
                    act = to_numpy(result.act)
                    obs_next, rew, done, info = env.step(act[0])
                    if done:
                        break
                    batch = Batch(state={}, obs=obs_next, act={}, rew={}, done={}, info={},
                                  obs_next={}, policy={})
            atk_dam = self.dam(batch.obs[-1]) if self.dam is not None else rew
            if abs(atk_dam - std_dam) > self.delta and atk_dam > best_dam:
                best_dam = atk_dam
                adv_acts = acts
                attack = True
                if not self.full_search:
                    return adv_acts, attack
        # print(std_dam, best_dam)
        return adv_acts, attack


def dam_pong(obs):
    """
    :param obs: (84 x 84) numpy array
    :return: int, pong observation dam value
    """
    obs = obs.astype(int)
    ball_obs = obs[14:-7, 11:-11]
    right_bar_obs = obs[14:-7, -11:]
    right_bar = 117
    empty = 87

    try:
        if not np.all(ball_obs == empty):
            shape_ball = np.argwhere(ball_obs != empty)
        else:  # ball already passed left or right bar
            return np.inf
        pos_ball = shape_ball[-1]
        pos_ball_yx = [pos_ball[0], pos_ball[1]]
        shape_right_bar = np.argwhere(right_bar_obs == right_bar)
        pos_right_bar = shape_right_bar[len(shape_right_bar) // 2]
        pos_right_bar_yx = [pos_right_bar[0], pos_right_bar[1]]
        dam = abs(pos_ball_yx[0] - pos_right_bar_yx[0])
        return dam
    except:  # error
        return 0


def dam_pacman(obs):
    """
    :param obs: (84 x 84) numpy array
    :return: int, pacman observation dam value
    """


def dam_enduro(obs):
    """
    :param obs: (84 x 84) numpy array
    :return: int, enduro observation dam value
    """


def dam_breakout(obs):
    """
    :param obs: (84 x 84) numpy array
    :return: int, breakout observation dam value
    """

    obs = obs.astype(int)[38:78, 5:-5]
    ball_obs = obs[:36]
    bar_obs = obs[36:]
    empty = 0

    try:
        if not np.all(ball_obs == empty):
            shape_ball = np.argwhere(ball_obs != empty)
        else:  # ball already passed the bar
            return np.inf
        pos_ball = shape_ball[-1]
        pos_ball_yx = [pos_ball[0], pos_ball[1]]
        shape_bar = np.argwhere(bar_obs != empty)
        pos_bar = shape_bar[len(shape_bar) // 2]
        pos_bar_yx = [pos_bar[0], pos_bar[1]]
        dam = abs(pos_ball_yx[1] - pos_bar_yx[1])
        return dam
    except:  # error
        return 0


def dam_cartpole(obs):
    """
    :param obs: (4,) numpy array
    :return: int, cartpole observation dam value
    """
    obs = obs.reshape(4)
    return abs(obs[2] * obs[3])  # inclination * angular velocity

