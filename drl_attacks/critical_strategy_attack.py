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
import itertools, copy
from tianshou.exploration import BaseNoise


class critical_strategy_attack_collector(Collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param obs_adv_atk: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack. It has to support
        targeted attacks.
    :param n: int, number of attacks in a sequence.
    :param m: int, number of observations in a sequence.
    :param beta: float, minimum reward margin to consider an adversarial
        sequence respect to the standard sequence.
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
                 obs_adv_atk: Attack,
                 n: int = 3,
                 m: int = None,
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
        if not perfect_attack:
            self.adv.targeted = True
        self.n = n
        if m == None:
            m = n
        self.m = m
        assert n <= m, \
            "n should be <= m"
        self.beta = beta
        assert beta >= 0, \
            "beta should be >= 0"
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

        def adversarial_policy(batch):
            """Find an adversarial policy or return the standard one.
            Return an array containing the adversarial actions and their length"""
            action_shape = self.env.action_space.shape or self.env.action_space.n
            action_shape = np.prod(action_shape)  # number of actions
            actions = [a for a in range(action_shape)]
            atk_strategies = [p for p in itertools.product(actions, repeat=self.n)]  # define attack strategies
            init_env = self.env
            env = copy.deepcopy(init_env)  # save deep copy of initial env
            ### test standard policy ###
            std_rew = 0  # cumulative reward
            best_acts = []  # actions of the best adversarial policy
            for i in range(self.m):
                with torch.no_grad():
                    result = self.policy(batch, None)
                act = to_numpy(result.act)
                obs_next, rew, done, info = env.step(act[0])
                obs = self._make_batch(obs_next)
                rew = self._make_batch(rew)
                std_rew += rew
                best_acts.append(act[0])
                if done:
                    break
                batch = Batch(
                    obs=obs, act=None, rew=None,
                    done=None, obs_next=None, info=None, policy=None)
            worst_atk_rew = std_rew  # best adversarial reward
            ### test adversarial policies ###
            for atk in atk_strategies:
                env = copy.deepcopy(init_env)
                acts = list(atk)
                atk_rew = 0
                for act in acts:  # play n steps according to adversarial policy
                    obs_next, rew, done, info = env.step(act)
                    atk_rew += rew
                    if done:
                        break
                if self.m > self.n and not done:  # play n-m steps according to standard policy
                    obs = self._make_batch(obs_next)
                    batch = Batch(
                        obs=obs, act=None, rew=None,
                        done=None, obs_next=None, info=None, policy=None)
                    for i in range(self.m - self.n):
                        with torch.no_grad():
                            result = self.policy(batch, None)
                        act = to_numpy(result.act)
                        obs_next, rew, done, info = env.step(act[0])
                        obs = self._make_batch(obs_next)
                        rew = self._make_batch(rew)
                        atk_rew += rew
                        if done:
                            break
                        batch = Batch(
                            obs=obs, act=None, rew=None,
                            done=None, obs_next=None, info=None, policy=None)
                if atk_rew + self.beta < std_rew and atk_rew < worst_atk_rew:
                    worst_atk_rew = atk_rew
                    best_acts = acts
            return best_acts

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
            if frames_count % self.m == 0:
                adv_acts = adversarial_policy(self.data)  # define adversarial policy
                count_n = self.n
                # print("Adv actions", adv_acts)
                # print("Lenght", len(adv_acts))
            if len(adv_acts) > 0:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                ori_obs = torch.FloatTensor(self.data.obs).to(device)  # get the original observations
                adv_act = adv_acts.pop(0)
                if not self.perfect_attack and count_n > 0:
                    n_attacks += 1
                    adv_act_t = torch.from_numpy(np.array([adv_act])).to(device)
                    adv_obs = self.adv.perturb(ori_obs, adv_act_t)  # create adversarial observations
                    y = self.adv.predict(adv_obs)
                    _, adv_actions = torch.max(y, 1)  # predict adversarial actions
                    self.data.act = adv_actions.cpu().detach().numpy()  # replace original actions with adversarial actions"""
                    count_n -= 1
                else:
                    self.data.act = [adv_act]
                    if self.perfect_attack:
                        n_attacks += 1
                if self.data.act == [adv_act] and count_n > 0:
                    succ_atk += 1
            frames_count += 1
            #####################################

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
            'n/ep': cur_episode,
            'n/st': cur_step,
            'v/st': self.step_speed.get(),
            'v/ep': self.episode_speed.get(),
            'rew': reward_sum,
            'len': length_sum / n_episode,
            'n_atks': n_attacks / np.sum(n_episode),
            'n_succ_atks': succ_atk / np.sum(n_episode),
            'atk_rate(%)': self.n / self.m,
            'succ_atks(%)': succ_atk / n_attacks if n_attacks > 0 else 0,
        }
