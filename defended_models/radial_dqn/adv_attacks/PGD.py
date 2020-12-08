from __future__ import division

import logging
from collections import Iterable

import numpy as np

from adv_attacks.base import Attack

class PGDAttack(Attack):
    """
    This class implements gradient attack method, and is the base of FGSM, BIM,
    ILCM, etc.
    """

    def __init__(self, model, support_targeted=True):
        """
        :param model(model): The model to be attacked.
        :param support_targeted(bool): Does this attack method support targeted.
        """
        super(PGDAttack, self).__init__(model)
        self.support_targeted = support_targeted

    def _apply(self,
               adversary,
               norm_ord = np.inf,
               epsilon = 0.01,
               steps = 10,
               relative_step_size = 0.2):
        """
        Apply the gradient attack method.
        :param adversary(Adversary):
            The Adversary object.
        :param norm_ord(int):
            Order of the norm, such as np.inf, 1, 2, etc. It can't be 0.
        :param epsilons(list|tuple|int):
            Attack step size (input variation).
            Largest step size if epsilons is not iterable.
        :param steps:
            The number of attack iteration.
        :param epsilon_steps:
            The number of Epsilons' iteration for each attack iteration.
        :return:
            adversary(Adversary): The Adversary object.
        """
            
        if norm_ord != np.inf:
            raise ValueError("only linf norm is supported!")

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        logging.info('epsilon={0},steps={1}'.
                     format(epsilon,steps))

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()
        #project to correct space
        min_ = np.maximum(min_, adversary.original-epsilon)
        max_ = np.minimum(max_, adversary.original+epsilon)

        #assert self.model.channel_axis() == adversary.original.ndim
        assert (self.model.channel_axis() == 1 or
                self.model.channel_axis() == adversary.original.shape[0] or
                self.model.channel_axis() == adversary.original.shape[-1])



        step = 1
        adv_img = np.copy(adversary.original)
        if epsilon == 0.0:
            adversary.try_accept_the_example(adv_img, adv_label)
            return adversary
        for i in range(steps):
            if adversary.is_targeted_attack:
                gradient = +self.model.gradient(adv_img,
                                                adversary.target_label)
            else:
                gradient = -self.model.gradient(adv_img,
                                               adversary.original_label)
            if norm_ord == np.inf:
                gradient_norm = np.sign(gradient)
            else:
                gradient_norm = gradient / self._norm(
                    gradient, ord=norm_ord)

            adv_img = adv_img + epsilon * relative_step_size * gradient_norm

            adv_img = np.clip(adv_img, min_, max_)
            step += 1
            
        
        adv_label = np.argmax(self.model.predict(adv_img))
        logging.info('step={}, epsilon = {:.5f}, pre_label = {}, adv_label={} logits={}'.
                     format(step, epsilon, pre_label,adv_label,self.model.predict(adv_img)[adv_label]))
        
        adversary.try_accept_the_example(adv_img, adv_label)
            
        return adversary

    @staticmethod
    def _norm(a, ord):
        if a.ndim == 1:
            return np.linalg.norm(a, ord=ord)
        if a.ndim == a.shape[0]:
            norm_shape = (a.ndim, reduce(np.dot, a.shape[1:]))
            norm_axis = 1
        else:
            norm_shape = (reduce(np.dot, a.shape[:-1]), a.ndim)
            norm_axis = 0
        return np.linalg.norm(a.reshape(norm_shape), ord=ord, axis=norm_axis)