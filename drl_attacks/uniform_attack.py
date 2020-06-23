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


class uniform_attack_collector(Collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param adv: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack.
    :param atk_frequency: float between 0 and 1, frequency of the attacks
        (i.e. 0.25 -> attack once each 4 frames)
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
                 env: Union[gym.Env, BaseVectorEnv],
                 adv: Attack,
                 atk_frequency: float = 1.,
                 perfect_attack: bool = False,
                 buffer: Optional[Union[ReplayBuffer, List[ReplayBuffer]]] = None,
                 preprocess_fn: Callable[[Any], Union[dict, Batch]] = None,
                 stat_size: Optional[int] = 100,
                 **kwargs
                 ):
        super().__init__(policy, env, buffer, preprocess_fn, stat_size, **kwargs)
        self.adv = adv  # advertorch attack method
        self.atk_frequency = atk_frequency
        assert 0 <= atk_frequency <= 1, \
            "atk_frequency should be included between 0 and 1"
        self.atk_frames = int(1 / atk_frequency)
        self.perfect_attack = perfect_attack

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
            * ``succ_atks(%)`` ratio of successful attacks over performed attacks.
        """
        warning_count = 0  # counts the number of steps
        if not self._multi_env:
            n_episode = np.sum(n_episode)
        start_time = time.time()
        assert sum([(n_step != 0), (n_episode != 0)]) == 1, \
            "One and only one collection number specification is permitted!"
        cur_step = 0
        cur_episode = np.zeros(self.env_num) if self._multi_env else 0
        reward_sum = 0
        length_sum = 0
        frames_count = 0  # number of observed frames
        n_attacks = 0  # number of attacks performed
        succ_atk = 0  # number of successful image attacks
        while True:
            if warning_count >= 100000:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)
            batch = Batch(
                obs=self._obs, act=self._act, rew=self._rew,
                done=self._done, obs_next=None, info=self._info,
                policy=None)
            if random:  # take random actions
                action_space = self.env.action_space
                if isinstance(action_space, list):
                    result = Batch(act=[a.sample() for a in action_space])
                else:
                    result = Batch(act=self._make_batch(action_space.sample()))
            else:  # take actions according to a policy
                with torch.no_grad():
                    result = self.policy(batch, self.state)
            self.state = result.get('state', None)  # get the state
            self._policy = to_numpy(result.policy) \
                if hasattr(result, 'policy') else [{}] * self.env_num  # distribution over actions
            self._act = to_numpy(result.act)
            ##########ADVERSARIAL ATTACK#########
            if frames_count % self.atk_frames == 0:
                if not self.perfect_attack:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    ori_obs = torch.FloatTensor(batch.obs).to(device)  # get the original observations
                    ori_act = self._act  # get the original actions
                    ori_act_t = torch.tensor(ori_act).to(device)
                    adv_obs = self.adv.perturb(ori_obs, ori_act_t)  # create adversarial observations
                    y = self.adv.predict(adv_obs)
                    _, adv_actions = torch.max(y, 1)  # predict adversarial actions
                    self._act = adv_actions.cpu().detach().numpy()  # replace original actions with adversarial actions
                else:
                    ori_act = self._act
                    action_shape = self.env.action_space.shape or self.env.action_space.n
                    while self._act == ori_act:
                        self._act = [rd.randint(0, np.prod(action_shape)-1)]
                if self._act != ori_act:
                    succ_atk += 1
                n_attacks += 1
            frames_count += 1
            ####################################
            obs_next, self._rew, self._done, self._info = self.env.step(
                self._act if self._multi_env else self._act[0])  # execute the actions
            if not self._multi_env:
                obs_next = self._make_batch(obs_next)
                self._rew = self._make_batch(self._rew)
                self._done = self._make_batch(self._done)
                self._info = self._make_batch(self._info)
            if log_fn:
                log_fn(self._info if self._multi_env else self._info[0])
            if render:
                self.env.render()
                if render > 0:
                    time.sleep(render)
            self.length += 1
            self.reward += self._rew
            if self.preprocess_fn:
                result = self.preprocess_fn(
                    obs=self._obs, act=self._act, rew=self._rew,
                    done=self._done, obs_next=obs_next, info=self._info,
                    policy=self._policy)
                self._obs = result.get('obs', self._obs)
                self._act = result.get('act', self._act)
                self._rew = result.get('rew', self._rew)
                self._done = result.get('done', self._done)
                obs_next = result.get('obs_next', obs_next)
                self._info = result.get('info', self._info)
                self._policy = result.get('policy', self._policy)
            if self._multi_env:
                for i in range(self.env_num):
                    data = {
                        'obs': self._obs[i], 'act': self._act[i],
                        'rew': self._rew[i], 'done': self._done[i],
                        'obs_next': obs_next[i], 'info': self._info[i],
                        'policy': self._policy[i]}
                    if self._cached_buf:
                        warning_count += 1
                        self._cached_buf[i].add(**data)
                    elif self._multi_buf:
                        warning_count += 1
                        self.buffer[i].add(**data)
                        cur_step += 1
                    else:
                        warning_count += 1
                        if self.buffer is not None:
                            self.buffer.add(**data)
                        cur_step += 1
                    if self._done[i]:
                        if n_step != 0 or np.isscalar(n_episode) or \
                                cur_episode[i] < n_episode[i]:
                            cur_episode[i] += 1
                            reward_sum += self.reward[i]
                            length_sum += self.length[i]
                            if self._cached_buf:
                                cur_step += len(self._cached_buf[i])
                                if self.buffer is not None:
                                    self.buffer.update(self._cached_buf[i])
                        self.reward[i], self.length[i] = 0, 0
                        if self._cached_buf:
                            self._cached_buf[i].reset()
                        self._reset_state(i)
                if sum(self._done):
                    obs_next = self.env.reset(np.where(self._done)[0])
                    if self.preprocess_fn:
                        obs_next = self.preprocess_fn(obs=obs_next).get(
                            'obs', obs_next)
                if n_episode != 0:
                    if isinstance(n_episode, list) and \
                            (cur_episode >= np.array(n_episode)).all() or \
                            np.isscalar(n_episode) and \
                            cur_episode.sum() >= n_episode:
                        break
            else:
                if self.buffer is not None:
                    self.buffer.add(
                        self._obs[0], self._act[0], self._rew[0],
                        self._done[0], obs_next[0], self._info[0],
                        self._policy[0])
                cur_step += 1
                if self._done:
                    cur_episode += 1
                    reward_sum += self.reward[0]
                    length_sum += self.length
                    self.reward, self.length = 0, 0
                    self.state = None
                    obs_next = self._make_batch(self.env.reset())
                    if self.preprocess_fn:
                        obs_next = self.preprocess_fn(obs=obs_next).get(
                            'obs', obs_next)
                if n_episode != 0 and cur_episode >= n_episode:
                    break
            if n_step != 0 and cur_step >= n_step:
                break
            self._obs = obs_next
        self._obs = obs_next
        if self._multi_env:
            cur_episode = sum(cur_episode)
        duration = max(time.time() - start_time, 1e-9)
        self.step_speed.add(cur_step / duration)
        self.episode_speed.add(cur_episode / duration)
        self.collect_step += cur_step
        self.collect_episode += cur_episode
        self.collect_time += duration
        if n_attacks == 0:
            n_attacks = 1e-9  # avoid 0 division
        if isinstance(n_episode, list):
            n_episode = np.sum(n_episode)
        else:
            n_episode = max(cur_episode, 1)
        return {
            'n/ep': cur_episode,
            'n/st': cur_step,
            'v/st': self.step_speed.get(),
            'v/ep': self.episode_speed.get(),
            'rew': reward_sum / n_episode,
            'len': length_sum / n_episode,
            'succ_atks(%)': succ_atk / n_attacks,
        }
