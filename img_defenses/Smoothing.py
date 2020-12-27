import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def scale(im, nR, nC):
    nR0 = len(im)  # source number of rows
    nC0 = len(im[0])  # source number of columns
    if nR == nR0 and nC == nC0:
        return im
    return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]


def SmoothingDefense(policy, smoothing="median", kernel_size=3, rescale=None):
    """
    Defend a policy applying image smoothing to input observations
    input obs should be [batch_size x n_channels x h x w]
    - example of policy: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/dqn.py
    Usage example:
        policy = load_policy(...)
        def_policy = SmoothingDefense(policy)

    :param policy: (policy) Tianshou Policy object
    :param smoothing: smoothing filter, can be ["median", "gaussian"]
    :param kernel_size: filter kernel size.
    :param rescale: float, whether to rescale images before applying filters,
    images that are too small may result too blurred even for small filters,
    hence they should be up-scaled and then down-scaled, operation which would tak a long time.
    :return: defended policy
    """

    f = policy.forward
    smoothing_type = ["median", "gaussian"]
    if smoothing not in smoothing_type:
        raise Exception("Smoothing algorithm not supported")

    def gaussian_filter_forward(s, last_state=None):
        """
        :param s: Tianshou Batch object
        :return: defended policy.forward method
        """
        x = s.obs
        for batch in range(x.shape[0]):
            for frame in range(x.shape[1]):
                x_ = x[batch][frame]
                if rescale:
                    x_ = ndimage.zoom(x[batch][frame], rescale)
                x_ = gaussian_filter(x_, sigma=kernel_size, order=0)
                if rescale:
                    x_ = ndimage.zoom(x_, 1 / rescale)
                x[batch][frame] = x_
        s.obs = x
        return f(s, last_state)

    def median_filter_forward(s, last_state=None):
        """
        :param s: Tianshou Batch object
        :return: defended policy.forward method
        """
        x = s.obs
        for batch in range(x.shape[0]):
            for frame in range(x.shape[1]):
                x_ = x[batch][frame]
                if rescale:
                    x_ = ndimage.zoom(x[batch][frame], rescale)
                x_ = median_filter(x_, size=kernel_size)
                if rescale:
                    x_ = ndimage.zoom(x_, 1 / rescale)
                x[batch][frame] = x_
        s.obs = x
        return f(s, last_state)

    if smoothing == "median":
        policy.forward = median_filter_forward
    elif smoothing == "gaussian":
        policy.forward = gaussian_filter_forward
    return policy


if __name__ == '__main__':
    from scipy import misc, ndimage
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ascent = misc.ascent()
    print(np.array(ascent).shape)
    result = median_filter(ascent, size=2)
    ax1.imshow(ascent)
    ax2.imshow(result)
    plt.show()
