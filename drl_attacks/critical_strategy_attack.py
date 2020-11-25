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


class critical_strategy_attack_collector(base_attack_collector):
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
    :param delta: float, minimum reward margin to consider an adversarial
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
                 full_search: bool = True
                 ):
        super().__init__(
            policy, env, obs_adv_atk, perfect_attack, device)

        if self.obs_adv_atk is not None:
            self.obs_adv_atk.targeted = True
        self.n = int(n)
        if not m:
            m = int(n)
        self.m = int(m)
        assert n <= m, \
            "n should be <= m"
        self.delta = delta
        assert delta >= 0, \
            "delta should be >= 0"
        self.acts_mask = acts_mask
        self.repeat_adv_act = repeat_adv_act
        assert repeat_adv_act >= 1, \
            "repeat_adv_act should be >= 1"
        self.m *= self.repeat_adv_act
        self.n *= self.repeat_adv_act
        self.env_state = None
        self.atari = atari
        self.count_n = 0
        self.count_m = 0
        self.full_search = full_search

    def store_env_state(self, env):
        """Copy the state of env in self.env_state"""
        if not self.atari:
            self.env_state = copy.deepcopy(env)
        else:
            # store state of atari env
            self.env_state = env.ale.cloneState()

    def load_env_state(self, env):
        """Copy the state of self.env_state in env and return it"""
        if not self.atari:
            env = self.env_state
        else:
            # load state of atari env
            env.ale.restoreState(self.env_state)
        return env

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
        std_rew = 0  # cumulative reward
        attack = False
        for i in range(self.m):
            with torch.no_grad():
                batch.obs = np.expand_dims(batch.obs, axis=0)
                result = self.policy(batch, None)
            act = to_numpy(result.act)
            obs_next, rew, done, info = env.step(act[0])
            std_rew += rew
            adv_acts.append(act[0])
            if done:
                break
            batch = Batch(state={}, obs=obs_next, act={}, rew={}, done={}, info={},
                          obs_next={}, policy={})
        lowest_rew = std_rew  # best adversarial reward
        # test adversarial policies
        for atk in atk_strategies:
            self.store_env_state(self.env)  # store the state of the environment
            env = self.load_env_state(env)  # restore the state of the environment
            acts = list(atk)
            atk_rew = 0
            for act in acts:  # play n steps according to adversarial policy
                obs_next, rew, done, info = env.step(act)
                atk_rew += rew
                if done:
                    break
            if self.m > self.n and not done:  # play n-m steps according to standard policy
                batch = Batch(state={}, obs=obs_next, act={}, rew={}, done={}, info={},
                              obs_next={}, policy={})
                for i in range(self.m - self.n):
                    with torch.no_grad():
                        batch.obs = np.expand_dims(batch.obs, axis=0)
                        result = self.policy(batch, None)
                    act = to_numpy(result.act)
                    obs_next, rew, done, info = env.step(act[0])
                    atk_rew += rew
                    if done:
                        break
                    batch = Batch(state={}, obs=obs_next, act={}, rew={}, done={}, info={},
                                  obs_next={}, policy={})
            if abs(atk_rew - std_rew) > self.delta and atk_rew < lowest_rew:
                lowest_rew = atk_rew
                adv_acts = acts
                attack = True
                if not self.full_search:
                    return adv_acts, attack
        # print(std_rew, lowest_rew)
        return adv_acts, attack

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
            self.count_n = 0
            self.count_m = 0
        self.data.obs = self.data.obs_next

    def collect(self,
                n_step: int = 0,
                n_episode: int = 0,
                render: Optional[float] = None
                ) -> Dict[str, float]:

        assert (n_step and not n_episode) or (not n_step and n_episode), \
            "One and only one collection number specification is permitted!"
        self.reset_env()
        self.reset_attack()
        self.count_m = 0
        self.count_n = 0

        while True:
            if render:
                self.render()
                time.sleep(render)
            self.show_warning()
            self.predict_next_action()

            # START ADVERSARIAL ATTACK
            if self.count_m == 0:
                adv_acts, attack = self.adversarial_policy(self.data)  # define adversarial policy
                self.count_n = len(adv_acts) if attack else 0
                self.count_m = self.m
                # print("Adv actions", adv_acts)
                # print("Lenght", len(adv_acts))
                # print(self.reward_total)
            if len(adv_acts) > 0 and self.count_n > 0:
                adv_act = adv_acts.pop(0)
                if not self.perfect_attack:
                    self.obs_attacks([adv_act])
                else:
                    self.data.act = [adv_act]
                if self.data.act == adv_act:
                    self.succ_attacks += 1
                self.n_attacks += 1
                self.count_n -= 1
            self.count_m -= 1
            self.frames_count += 1
            # END ADVERSARIAL ATTACK

            self.perform_step()
            if self.check_end_attack(n_step, n_episode):
                break

        return self.get_attack_stats()
