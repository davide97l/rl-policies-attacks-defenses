from tianshou.data import Collector
from advertorch.attacks.base import Attack
import gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer, to_numpy
import random as rd
from tianshou.exploration import BaseNoise
import torch.nn as nn


class adversarial_policy_attack_collector(Collector, base_attack_collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class. Policy defining the adversarial strategy.
    :param victim_policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class. Policy we want to attack.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param obs_adv_atk: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack.
    :param beta: attacks only if max(prob actions) - min(prob actions) >= beta.
    :param perfect_attack: force adversarial attacks on observations to be
        always effective (ignore the ``adv`` param).
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer`
        class, or a list of :class:`~tianshou.data.ReplayBuffer`. If set to
        ``None``, it will automatically assign a small-size
        :class:`~tianshou.data.ReplayBuffer`.
    :param function preprocess_fn: a function called before the data has been
        added to the buffer, see issue #42, defaults to ``None``.
    :param int stat_size: for the moving average of recording speed, defaults
        to 100.
    The ``preprocess_fn`` is a function called before the data has been added
    to the buffer with batch format, which receives up to 7 keys as listed in
    :class:`~tianshou.data.Batch`. It will receive with only ``obs`` when the
    collector resets the environment. It returns either a dict or a
    :class:`~tianshou.data.Batch` with the modified keys and values. Examples
    are in "test/base/test_collector.py".
    """

    def __init__(self,
                 policy: BasePolicy,
                 victim_policy: BasePolicy,
                 env: Union[gym.Env, BaseVectorEnv],
                 obs_adv_atk: Attack,
                 beta: float = 0.,
                 perfect_attack: bool = False,
                 buffer: Optional[Union[ReplayBuffer, List[ReplayBuffer]]] = None,
                 preprocess_fn: Callable[[Any], Union[dict, Batch]] = None,
                 action_noise: Optional[BaseNoise] = None,
                 reward_metric: Optional[Callable[[np.ndarray], float]] = None,
                 **kwargs
                 ):
        super().__init__(policy, env, buffer, preprocess_fn, action_noise, reward_metric, **kwargs)
        self.adv = obs_adv_atk  # advertorch attack method
        self.perfect_attack = perfect_attack
        self.victim_policy = victim_policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.beta = beta
        self.action_space = self.env.action_space[0].shape or self.env.action_space[0].n

    def collect(self,
                n_step: int = 0,
                n_episode: Union[int, List[int]] = 0,
                random: bool = False,
                render: Optional[float] = None,
                log_fn: Optional[Callable[[dict], None]] = None
                ) -> Dict[str, float]:
        """Collect a specified number of step or episode.
        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect (in each
            environment).
        :type n_episode: int or list
        :param bool random: whether to use random policy for collecting data,
            defaults to ``False``.
        :param float render: the sleep time between rendering consecutive
            frames, defaults to ``None`` (no rendering).
        :param function log_fn: a function which receives env info, typically
            for tensorboard logging.
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
        if not self._multi_env:
            n_episode = np.sum(n_episode)
        start_time = time.time()
        assert sum([(n_step != 0), (n_episode != 0)]) == 1, \
            "One and only one collection number specification is permitted!"
        cur_step, cur_episode = 0, np.zeros(self.env_num)
        reward_sum, length_sum = 0., 0

        frames_count = 0  # number of observed frames
        n_attacks = 0  # number of attacks performed
        succ_atk = 0  # number of successful image attacks
        n_attacks_ep = 0  # number of attacks performed in the same episode

        while True:
            if cur_step >= 100000 and cur_episode.sum() == 0:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)

            # restore the state and the input data
            last_state = self.data.state
            if last_state.is_empty():
                last_state = None
            self.data.update(state=Batch(), obs_next=Batch(), policy=Batch())

            # calculate the next action
            if random:
                action_space = self.env.action_space
                if isinstance(action_space, list):
                    result = Batch(act=[a.sample() for a in action_space])
                else:
                    result = Batch(act=self._make_batch(action_space.sample()))
            else:
                with torch.no_grad():
                    result = self.policy(self.data, last_state)

            # convert None to Batch(), since None is reserved for 0-init
            state = result.get('state', Batch())
            if state is None:
                state = Batch()
            self.data.state = state
            if hasattr(result, 'policy'):
                self.data.policy = to_numpy(result.policy)
            # save hidden state to policy._state, in order to save into buffer
            self.data.policy._state = self.data.state

            self.data.act = to_numpy(result.act)  # a'
            if self._action_noise is not None:
                self.data.act += self._action_noise(self.data.act.shape)

            ##########ADVERSARIAL ATTACK#########
            softmax = nn.Softmax(dim=1)
            prob_a = softmax(result.logits).cpu().detach().numpy()  # distribution over actions
            max_a = np.amax(prob_a)
            min_a = np.amin(prob_a)
            diff = max_a - min_a

            adv_act = self.data.act
            attacked = False
            if not self.perfect_attack:
                # if n_attacks_ep < self.n:
                if diff >= self.beta:
                    ori_obs = torch.FloatTensor(self.data.obs).to(self.device)  # get the original observations
                    adv_act_t = torch.tensor(self.data.act).to(self.device)
                    adv_obs = self.adv.perturb(ori_obs, adv_act_t)  # s'_t
                    n_attacks += 1
                    n_attacks_ep += 1
                    attacked = True
                else:
                    adv_obs = self.data.obs  # s'_t
                batch = Batch(obs=adv_obs, info=None)
                with torch.no_grad():
                    result = self.victim_policy(batch, None)
                self.data.act = result.act  # a''
                if self.data.act == adv_act and attacked:
                    succ_atk += 1
            else:
                if diff >= self.beta:
                    succ_atk += 1
                    n_attacks += 1
                else:
                    batch = Batch(obs=self.data.obs, info=None)
                    with torch.no_grad():
                        result = self.victim_policy(batch, None)
                    self.data.act = result.act  # a''
            frames_count += 1
            ####################################

            # step in env
            obs_next, rew, done, info = self.env.step(self.data.act)
            rew = -rew  # reward of the adv_agent is the negative of the reward of the victim agent
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
            'n/ep': cur_episode,
            'n/st': cur_step,
            'v/st': self.step_speed.get(),
            'v/ep': self.episode_speed.get(),
            'rew': -reward_sum,
            'len': length_sum / n_episode,
            'n_atks': n_attacks / np.sum(n_episode),
            'n_succ_atks': succ_atk / np.sum(n_episode),
            'atk_rate(%)': n_attacks / frames_count,
            'succ_atks(%)': succ_atk / n_attacks if n_attacks > 0 else 0,
        }