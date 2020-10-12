import torch.nn as nn
from advertorch.attacks.base import Attack
from drl_attacks.base_attack import base_attack_collector
import gym
import time
import torch
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
from tianshou.policy import BasePolicy
from net.discrete_net import DQN
from tianshou.policy import DQNPolicy
from tianshou.data import Batch, to_numpy


class adversarial_policy_attack_collector(base_attack_collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param obs_adv_atk: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack.
    :param beta: attacks only if max(prob actions) - min(prob actions) >= beta
    :param softmax: if true, apply softmax to convert logits into probabilities
        during observation evaluation
    :param perfect_attack: force adversarial attacks on observations to be
        always effective (ignore the ``adv`` param).
    :param adv_policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class. Defines the behavior of the adversarial policy. If None it will
        be replaced by a random DQN policy.
    """
    def __init__(self,
                 policy: BasePolicy,
                 env: gym.Env,
                 obs_adv_atk: Attack,
                 perfect_attack: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 beta: float = 0.0,
                 softmax: bool = True,
                 adv_policy: BasePolicy = None
                 ):
        super().__init__(
            policy, env, obs_adv_atk, perfect_attack, device)

        self.beta = beta
        assert 0 <= beta, \
            "beta should >= 0"
        self.softmax = softmax
        if self.obs_adv_atk is not None:
            self.obs_adv_atk.targeted = True

        self.adv_policy = adv_policy
        if self.adv_policy is None:
            state_shape = env.observation_space.shape or env.observation_space.n
            action_shape = env.env.action_space.shape or env.env.action_space.n
            self.adv_net = DQN(*state_shape, action_shape, self.device).to(self.device)
            self.adv_policy = DQNPolicy(self.adv_net, None)

    def predict_adv_action(self):
        """
        Predicts the next adversarial action given observation 'self.data.obs' and policy 'self.policy',
        and stores it in 'self.data.act'
        :return: outcome of policy forward pass
        """
        temp_obs = self.data.obs
        with torch.no_grad():
            self.data.obs = np.expand_dims(self.data.obs, axis=0)
            result = self.adv_policy(self.data, last_state=None)
        self.data.obs = temp_obs
        self.data.adv_act = to_numpy(result.act)
        return result

    def collect(self,
                n_step: int = 0,
                n_episode: int = 0,
                render: Optional[float] = None
                ) -> Dict[str, float]:

        assert (n_step and not n_episode) or (not n_step and n_episode), \
            "One and only one collection number specification is permitted!"
        self.reset_env()
        self.reset_attack()

        while True:
            if render:
                self.render()
                time.sleep(render)
            self.show_warning()
            adv_result = self.predict_adv_action()
            result = self.predict_next_action()

            # START ADVERSARIAL ATTACK
            if self.softmax:
                softmax = nn.Softmax(dim=-1)
                prob_a = softmax(adv_result.logits).cpu().detach().numpy()  # distribution over actions
            else:
                prob_a = adv_result.logits.cpu().detach().numpy()  # action logits
            max_a = np.amax(prob_a)
            min_a = np.amin(prob_a)
            diff = max_a - min_a
            # print(max_a, min_a, diff)
            if diff >= self.beta:
                target_act = [int(np.argmax(prob_a))]  # get the desired action
                if not self.perfect_attack:
                    self.obs_attacks(target_act)
                else:
                    self.data.act = target_act
                if self.data.act == target_act:
                    self.succ_attacks += 1
                self.n_attacks += 1
            self.frames_count += 1
            # END ADVERSARIAL ATTACK

            self.perform_step()
            if self.check_end_attack(n_step, n_episode):
                break

        return self.get_attack_stats()
