from tianshou.data import Collector
from advertorch.attacks.base import Attack
import random as rd
import gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer, to_numpy


class uniform_attack_collector(Collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param obs_adv_atk: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack.
    :param atk_frequency: float between 0 and 1, frequency of the attacks
        (i.e. 0.25 -> attack once each 4 frames)
    :param perfect_attack: force adversarial attacks on observations to be
        always effective (ignore the ``adv`` param).
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer`
        class. If set to ``None`` (testing phase), it will not store the data.
    :param function preprocess_fn: a function called before the data has been
        added to the buffer, see issue #42 and :ref:`preprocess_fn`, defaults
        to ``None``.
    :param BaseNoise action_noise: add a noise to continuous action. Normally
        a policy already has a noise param for exploration in training phase,
        so this is recommended to use in test collector for some purpose.
    :param function reward_metric: to be used in multi-agent RL. The reward to
        report is of shape [agent_num], but we need to return a single scalar
        to monitor training. This function specifies what is the desired
        metric, e.g., the reward of agent 1 or the average reward over all
        agents. By default, the behavior is to select the reward of agent 1.

    The ``preprocess_fn`` is a function called before the data has been added
    to the buffer with batch format, which receives up to 7 keys as listed in
    :class:`~tianshou.data.Batch`. It will receive with only ``obs`` when the
    collector resets the environment. It returns either a dict or a
    :class:`~tianshou.data.Batch` with the modified keys and values. Examples
    are in "test/base/test_collector.py".
    """

    def __init__(self,
                 policy: BasePolicy,
                 env: Union[gym.Env, BaseVectorEnv],
                 obs_adv_atk: Attack,
                 atk_frequency: float = 1.,
                 perfect_attack: bool = False,
                 buffer: Optional[Union[ReplayBuffer, List[ReplayBuffer]]] = None,
                 preprocess_fn: Callable[[Any], Union[dict, Batch]] = None,
                 action_noise: Optional[BaseNoise] = None,
                 reward_metric: Optional[Callable[[np.ndarray], float]] = None,
                 ):
        super().__init__(policy, env, buffer, preprocess_fn, action_noise, reward_metric)
        self.adv = obs_adv_atk  # advertorch attack method
        self.atk_frequency = atk_frequency
        assert 0 <= atk_frequency <= 1, \
            "atk_frequency should be included between 0 and 1"
        self.atk_frequency = atk_frequency
        self.atk_frames = int(1 / atk_frequency)
        self.perfect_attack = perfect_attack
        self.action_space = self.env.action_space[0].shape or self.env.action_space[0].n

    def collect(self,
                n_step: int = 0,
                n_episode: Union[int, List[int]] = 0,
                random: bool = False,
                render: Optional[float] = None,
                device: str = 'cpu'
                ) -> Dict[str, float]:
        """Collect a specified number of step or episode.
        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect. If it is an
            int, it means to collect at lease ``n_episode`` episodes; if it is
            a list, it means to collect exactly ``n_episode[i]`` episodes in
            the i-th environment
        :param bool random: whether to use random policy for collecting data,
            defaults to ``False``.
        :param float render: the sleep time between rendering consecutive
            frames, defaults to ``None`` (no rendering).
        :param str device: can be 'cpu' or 'cuda'
        .. note::
            One and only one collection number specification is permitted,
            either ``n_step`` or ``n_episode``.
        :return: A dict including the following keys
            * ``n/ep`` the collected number of episodes.
            * ``n/st`` the collected number of steps.
            * ``v/st`` the speed of steps per second.
            * ``v/ep`` the speed of episode per second.
            * ``rew`` the mean reward over collected episodes.
            * ``len`` the mean length over collected episodes.
            * ``n_attacks`` number of performed attacks.
            * ``n_succ_atks`` number of performed successful attacks.
            * ``n_attacks(%)`` ratio of performed attacks over steps.
            * ``succ_atks(%)`` ratio of successful attacks over performed attacks.
        """
        assert (n_step and not n_episode) or (not n_step and n_episode), \
            "One and only one collection number specification is permitted!"
        start_time = time.time()
        step_count = 0
        # episode of each environment
        episode_count = np.zeros(self.env_num)
        reward_total = 0.0
        whole_data = Batch()

        frames_count = 0  # number of observed frames
        n_attacks = 0  # number of attacks performed
        succ_atk = 0  # number of successful image attacks

        while True:
            if step_count >= 100000 and episode_count.sum() == 0:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)

            # restore the state and the input data
            last_state = self.data.state
            if isinstance(last_state, Batch) and last_state.is_empty():
                last_state = None
            self.data.update(state=Batch(), obs_next=Batch(), policy=Batch())

            # calculate the next action
            if random:
                spaces = self._action_space
                result = Batch(
                    act=[spaces[i].sample() for i in self._ready_env_ids])
            else:
                with torch.no_grad():
                    result = self.policy(self.data, last_state)

            state = result.get('state', Batch())
            # convert None to Batch(), since None is reserved for 0-init
            if state is None:
                state = Batch()
            self.data.update(state=state, policy=result.get('policy', Batch()))
            # save hidden state to policy._state, in order to save into buffer
            if not (isinstance(self.data.state, Batch)
                    and self.data.state.is_empty()):
                self.data.policy._state = self.data.state

            self.data.act = to_numpy(result.act)
            if self._action_noise is not None:
                self.data.act += self._action_noise(self.data.act.shape)

            ##########ADVERSARIAL ATTACK#########
            if frames_count % self.atk_frames == 0:
                if not self.perfect_attack:
                    ori_obs = torch.FloatTensor(self.data.obs).to(device)  # get the original observations
                    ori_act = self.data.act  # get the original actions
                    ori_act_t = torch.tensor(ori_act).to(device)
                    adv_obs = self.adv.perturb(ori_obs, ori_act_t)  # create adversarial observations
                    y = self.adv.predict(adv_obs)
                    _, adv_actions = torch.max(y, 1)  # predict adversarial actions
                    self._act = adv_actions.cpu().detach().numpy()  # replace original actions with adversarial actions
                else:
                    ori_act = self.data.act
                    while self.data.act == ori_act:
                        self.data.act = [rd.randint(0, self.action_space - 1)]
                if self.data.act != ori_act:
                    succ_atk += 1
                n_attacks += 1
            frames_count += 1
            ####################################

            # step in env
            obs_next, rew, done, info = self.env.step(self.data.act)
            # move data to self.data
            self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)

            if render:
                self.render()
                time.sleep(render)

            # add data into the buffer
            if self.preprocess_fn:
                result = self.preprocess_fn(**self.data)
                self.data.update(result)
            for j, i in enumerate(self._ready_env_ids):
                # j is the index in current ready_env_ids
                # i is the index in all environments
                self._cached_buf[i].add(**self.data[j])
                if self.data.done[j]:
                    if n_step or np.isscalar(n_episode) or \
                            episode_count[i] < n_episode[i]:
                        episode_count[i] += 1
                        reward_total += np.sum(self._cached_buf[i].rew, axis=0)
                        step_count += len(self._cached_buf[i])
                        if self.buffer is not None:
                            self.buffer.update(self._cached_buf[i])
                    self._cached_buf[i].reset()
                    self._reset_state(j)
            obs_next = self.data.obs_next
            if sum(self.data.done):
                env_ind_local = np.where(self.data.done)[0]
                env_ind_global = self._ready_env_ids[env_ind_local]
                obs_reset = self.env.reset(env_ind_global)
                if self.preprocess_fn:
                    obs_next[env_ind_local] = self.preprocess_fn(
                        obs=obs_reset).get('obs', obs_reset)
                else:
                    obs_next[env_ind_local] = obs_reset
            self.data.obs = obs_next
            if n_step:
                if step_count >= n_step:
                    break
            else:
                if isinstance(n_episode, int) and \
                        episode_count.sum() >= n_episode:
                    break
                if isinstance(n_episode, list) and \
                        (episode_count >= n_episode).all():
                    break

        # generate the statistics
        episode_count = sum(episode_count)
        duration = max(time.time() - start_time, 1e-9)
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += duration
        # average reward across the number of episodes
        reward_avg = reward_total / episode_count
        if np.asanyarray(reward_avg).size > 1:  # non-scalar reward_avg
            reward_avg = self._rew_metric(reward_avg)
        return {
            'n/ep': episode_count,
            'n/st': step_count,
            'v/st': step_count / duration,
            'v/ep': episode_count / duration,
            'rew': reward_avg,
            'len': step_count / episode_count,
            'n_atks': n_attacks / np.sum(n_episode),
            'n_succ_atks': succ_atk / np.sum(n_episode),
            'atk_rate(%)': self.atk_frequency,
            'succ_atks(%)': succ_atk / n_attacks if n_attacks > 0 else 0,
        }