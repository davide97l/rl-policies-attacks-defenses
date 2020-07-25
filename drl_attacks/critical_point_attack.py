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


class critical_point_attack_collector(Collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param adv: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack. It has to support
        targeted attacks.
    :param n: int, number of attacks in a sequence.
    :param m: int, number of observations in a sequence.
    :param beta: float, minimum DAM margin to consider an adversarial
        sequence respect to the standard sequence.
    :param function dam: danger aware metrics, if None, DAM is based on the
        reward achieved at the current state. Higher DAM correspond to
        worst state.
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
                 n: int = 3,
                 m: int = None,
                 beta: float = 0.,
                 dam: Callable = None,
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
        self.dam = dam

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
            best_acts = []  # actions of the best adversarial policy
            for i in range(self.m):
                with torch.no_grad():
                    result = self.policy(batch, None)
                act = to_numpy(result.act)
                obs_next, rew, done, info = env.step(act[0])
                obs = self._make_batch(obs_next)
                best_acts.append(act[0])
                if done:
                    break
                batch = Batch(obs=obs, act=None, rew=None,
                              done=done, obs_next=None, info=None, policy=None)
            if self.dam is not None:
                std_dam = self.dam(obs)  # standard DAM
            else:
                std_dam = -rew  # reward-based DAM
            best_atk_dam = std_dam
            ### test adversarial policies ###
            for atk in atk_strategies:
                env = copy.deepcopy(init_env)  # copy initial environment state
                acts = list(atk)
                atk_len = 0
                for act in acts:  # play n steps according to adversarial policy
                    obs_next, rew, done, info = env.step(act)
                    if done:
                        break
                obs = self._make_batch(obs_next)
                if self.m > self.n and not done:  # play n-m steps according to standard policy
                    batch = Batch(obs=obs, act=None, rew=None,
                                  done=done, obs_next=None, info=None, policy=None)
                    for i in range(self.m - self.n):
                        with torch.no_grad():
                            result = self.policy(batch, None)
                        act = to_numpy(result.act)
                        obs_next, rew, done, info = env.step(act[0])
                        obs = self._make_batch(obs_next)
                        if done:
                            break
                        batch = Batch(obs=obs, act=None, rew=None,
                                      done=done, obs_next=None, info=None, policy=None)
                if self.dam is not None:
                    atk_dam = self.dam(obs)  # adversarial DAM
                else:
                    atk_dam = -rew
                if atk_dam + self.beta > std_dam and atk_dam > best_atk_dam:
                    best_atk_dam = atk_dam
                    best_acts = acts
            return best_acts

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
            obs_next, rew, done, info = self.env.step(
                self.data.act if self._multi_env else self.data.act[0])

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
            'rew': reward_sum,
            'len': length_sum / n_episode,
            'n_atks': n_attacks / np.sum(n_episode),
            'n_succ_atks': succ_atk / np.sum(n_episode),
            'atk_rate(%)': self.n / self.m,
            'succ_atks(%)': succ_atk / n_attacks,
        }
