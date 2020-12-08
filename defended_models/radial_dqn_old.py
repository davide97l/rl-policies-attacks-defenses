from tianshou.policy import DQNPolicy
from typing import Any, Dict, Union, Optional
import numpy as np
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from .ipb import network_bounds
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


class RadialDQNPolicy(DQNPolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.
    Implementation of Double Q-Learning. arXiv:1509.06461.
    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).
    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        k: float = 0.5,
        eps: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor, estimation_step,
                         target_update_freq, reward_normalization, **kwargs)
        self.k = k
        self.eps = eps
        self.device = device

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._cnt % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q_logits = self(batch).logits
        q = q_logits[np.arange(len(q_logits)), batch.act]
        r = to_torch_as(batch.returns.flatten(), q)
        td = r - q
        std_loss = (torch.min(td.pow(2), torch.abs(td)) * weight).mean()

        obs = torch.from_numpy(batch.obs).float().to(self.device)

        #upper, lower = network_bounds(self.model, obs, self.eps)

        # compute bounds with lirpa
        model = BoundedModule(self.model, obs)
        # Define perturbation
        ptb = PerturbationLpNorm(norm=np.inf, eps=0.01)
        # Make the input a BoundedTensor with perturbation
        my_input = BoundedTensor(obs, ptb)
        # Forward propagation using BoundedTensor
        prediction = model(my_input)
        # Compute LiRPA bounds
        lower, upper = model.compute_bounds(method="backward")

        onehot_labels = torch.zeros(upper.shape).to(self.device)
        onehot_labels[range(batch.obs.shape[0]), batch.act] = 1

        #for layer in self.model.modules():
        #    print(type(layer))

        r = torch.repeat_interleave(r, q_logits.shape[-1], dim=-1).reshape(r.shape[0], q_logits.shape[-1])
        upper_diff = upper - q_logits * (1 - onehot_labels) - r * onehot_labels
        lower_diff = lower - q_logits * (1 - onehot_labels) - r * onehot_labels

        wc_diff = torch.max(torch.abs(upper_diff), torch.abs(lower_diff))

        # sum over output layer, mean only in batch dimension
        worst_case_loss = torch.sum(torch.min(wc_diff.pow(2), wc_diff), dim=1).mean() * weight

        loss = (self.k * std_loss + (1 - self.k) * worst_case_loss)

        #print(worst_case_loss, std_loss, loss)

        batch.weight = td  # prio-buffer
        loss.backward()
        self.optim.step()
        self._cnt += 1
        return {"loss": loss.item()}
