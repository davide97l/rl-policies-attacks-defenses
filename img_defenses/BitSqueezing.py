import numpy as np


def BitSqueezingDefense(policy, bit_depth=5, vmin=0., vmax=255.):
    """
    Defend a policy applying bit-squeezing to input observations
    input obs should be [batch_size x n_channels x h x w]
    - example of policy: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/dqn.py
    Usage example:
        policy = load_policy(...)
        def_policy = BitSqueezingDefense(policy)

    :param policy: (policy) Tianshou Policy object
    :param bit_depth: bit depth.
    :param vmin: min value.
    :param vmax: max value.
    :return: defended policy
    """

    f = policy.forward
    max_int = 2 ** bit_depth - 1

    def bit_squeezing(x, max_int, vmin, vmax):
        # here converting x such that: 0 =< x =< 1
        x = (x - vmin) / (vmax - vmin)
        x = np.round(x * max_int) / max_int
        return x * (vmax - vmin) + vmin

    def defense_forward(s, last_state=None):
        """
        :param s: Tianshou Batch object
        :return: defended policy.forward method
        """

        x = s.obs
        for batch in range(x.shape[0]):
            for frame in range(x.shape[1]):
                x[batch][frame] = bit_squeezing(x[batch][frame], max_int, vmin, vmax)
        s.obs = x
        return f(s, last_state)

    policy.forward = defense_forward
    return policy
