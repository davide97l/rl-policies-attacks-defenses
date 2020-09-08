from advertorch.attacks.base import Attack
from drl_attacks.base_attack import base_attack_collector
import random as rd
import gym
import time
import torch
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
from tianshou.policy import BasePolicy


class uniform_attack_collector(base_attack_collector):
    """
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param obs_adv_atk: an instance of the :class:`~advertorch.attacks.base.Attack`
        class implementing an image adversarial attack.
    :param perfect_attack: force adversarial attacks on observations to be
        always effective (ignore the ``adv`` param).
    :param atk_frequency: float, how frequently attacking env observations
    """
    def __init__(self,
                 policy: BasePolicy,
                 env: gym.Env,
                 obs_adv_atk: Attack,
                 perfect_attack: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 atk_frequency: float = 1.
                 ):
        super().__init__(
            policy, env, obs_adv_atk, perfect_attack, device)

        self.atk_frequency = atk_frequency
        if self.obs_adv_atk is not None:
            self.obs_adv_atk.targeted = False

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
            self.predict_next_action()

            # START ADVERSARIAL ATTACK
            x = rd.uniform(0, 1)
            if x < self.atk_frequency:
                ori_act = self.data.act
                if not self.perfect_attack:
                    self.obs_attacks(ori_act)
                else:
                    while self.data.act == ori_act:
                        self.data.act = [rd.randint(0, self.action_space - 1)]
                if self.data.act != ori_act:
                    self.succ_attacks += 1
                self.n_attacks += 1
            self.frames_count += 1
            # END ADVERSARIAL ATTACK

            self.perform_step()
            if self.check_end_attack(n_step, n_episode):
                break

        return self.get_attack_stats()
