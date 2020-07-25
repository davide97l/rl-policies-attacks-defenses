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


class antagonist_attack_collector(Collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class. Policy defining the adversarial strategy.
    :param victim_policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class. Policy we want to attack.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param adv: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack.
    :param n: int, maximum number of attacks per episode.
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
                 adv: Attack,
                 n: int = 10,
                 perfect_attack: bool = False,
                 buffer: Optional[Union[ReplayBuffer, List[ReplayBuffer]]] = None,
                 preprocess_fn: Callable[[Any], Union[dict, Batch]] = None,
                 stat_size: Optional[int] = 100,
                 action_noise: Optional[BaseNoise] = None,
                 reward_metric: Optional[Callable[[np.ndarray], float]] = None,
                 **kwargs
                 ):
        super().__init__(policy, env, buffer, preprocess_fn, stat_size, action_noise, reward_metric, **kwargs)
        self.adv = adv  # advertorch attack method
        self.perfect_attack = perfect_attack
        self.victim_policy = victim_policy
        self.n = n

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
            adv_act = self.data.act
            attacked = False
            if not self.perfect_attack:
                if n_attacks_ep < self.n:
                    ori_obs = torch.FloatTensor(self.data.obs).to(device)  # get the original observations
                    adv_act_t = torch.tensor(self.data.act).to(device)
                    adv_obs = self.adv.perturb(ori_obs, adv_act_t)  # s'_t
                    n_attacks += 1
                    n_attacks_ep += 1
                    attacked = True
                else:
                    adv_obs = self.obs  # s'_t
                batch = Batch(obs=adv_obs, info=None)
                with torch.no_grad():
                    result = self.victim_policy(batch, None)
                self.act = result.act  # a''
                if self.act == adv_act and attacked:
                    succ_atk += 1
            else:
                succ_atk += 1
                n_attacks += 1
            frames_count += 1
            ####################################

            # step in env
            obs_next, rew, done, info = self.env.step(
                self.data.act if self._multi_env else self.data.act[0])  # s_t+1, r_adv, done

            # move data to self.data
            if not self._multi_env:
                obs_next = self._make_batch(obs_next)
                rew = self._make_batch(rew)
                done = self._make_batch(done)
                info = self._make_batch(info)
            self.data.obs_next = obs_next
            self.data.rew = rew
            self.data.done = done
            self.data.info = info

            if log_fn:
                log_fn(info if self._multi_env else info[0])
            if render:
                self.render()
                if render > 0:
                    time.sleep(render)

            # add data into the buffer
            self.length += 1
            self.reward += self.data.rew
            if self.preprocess_fn:
                result = self.preprocess_fn(**self.data)
                self.data.update(result)
            if self._multi_env:  # cache_buffer branch
                for i in range(self.env_num):
                    self._cached_buf[i].add(**self.data[i])
                    if self.data.done[i]:
                        if n_step != 0 or np.isscalar(n_episode) or \
                                cur_episode[i] < n_episode[i]:
                            cur_episode[i] += 1
                            reward_sum += self.reward[i]
                            length_sum += self.length[i]
                            if self._cached_buf:
                                cur_step += len(self._cached_buf[i])
                                if self.buffer is not None:
                                    self.buffer.update(self._cached_buf[i])
                        self.reward[i], self.length[i] = 0., 0
                        if self._cached_buf:
                            self._cached_buf[i].reset()
                        self._reset_state(i)
                obs_next = self.data.obs_next
                if sum(self.data.done):
                    env_ind = np.where(self.data.done)[0]
                    obs_reset = self.env.reset(env_ind)
                    if self.preprocess_fn:
                        obs_next[env_ind] = self.preprocess_fn(
                            obs=obs_reset).get('obs', obs_reset)
                    else:
                        obs_next[env_ind] = obs_reset
                self.data.obs_next = obs_next
                if n_episode != 0:
                    if isinstance(n_episode, list) and \
                            (cur_episode >= np.array(n_episode)).all() or \
                            np.isscalar(n_episode) and \
                            cur_episode.sum() >= n_episode:
                        break
            else:  # single buffer, without cache_buffer
                if self.buffer is not None:
                    self.buffer.add(**self.data[0])
                cur_step += 1
                if self.data.done[0]:
                    cur_episode += 1
                    reward_sum += self.reward[0]
                    length_sum += self.length[0]
                    n_attacks_ep = 0  # reset counter
                    self.reward, self.length = 0., np.zeros(self.env_num)
                    self.data.state = Batch()
                    obs_next = self._make_batch(self.env.reset())
                    if self.preprocess_fn:
                        obs_next = self.preprocess_fn(obs=obs_next).get(
                            'obs', obs_next)
                    self.data.obs_next = obs_next
                if n_episode != 0 and cur_episode >= n_episode:
                    break
            if n_step != 0 and cur_step >= n_step:
                break
            self.data.obs = self.data.obs_next
        self.data.obs = self.data.obs_next

        # generate the statistics
        cur_episode = sum(cur_episode)
        duration = max(time.time() - start_time, 1e-9)
        self.step_speed.add(cur_step / duration)
        self.episode_speed.add(cur_episode / duration)
        self.collect_step += cur_step
        self.collect_episode += cur_episode
        self.collect_time += duration
        if isinstance(n_episode, list):
            n_episode = np.sum(n_episode)
        else:
            n_episode = max(cur_episode, 1)
        reward_sum /= n_episode
        if np.asanyarray(reward_sum).size > 1:  # non-scalar reward_sum
            reward_sum = self._rew_metric(reward_sum)
        if n_attacks == 0:
            n_attacks = 1e-9  # avoid 0 division
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
            'succ_atks(%)': succ_atk / n_attacks,
        }