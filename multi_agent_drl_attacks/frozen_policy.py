import numpy as np
from typing import Dict, Union, Optional
from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_numpy


class FrozenPolicy(BasePolicy):

    def __init__(self, policy: BasePolicy):
        self.__dict__ = policy.__dict__
        self.policy = policy

    def learn(self, batch: Batch, **kwargs):
        """This agent doesn't learn, so it returns an
        empty dict."""
        return {}

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:
        return self.policy.forward(batch, state, **kwargs)