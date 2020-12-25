from torchvision import transforms
from PIL import Image
import numpy as np

_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


def JPEGFilterDefense(policy, quality=10):
    """
    Defend a policy converting input observations to jpeg
    input obs should be [batch_size x n_channels x h x w]
    - example of policy: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/dqn.py
    Usage example:
        policy = load_policy(...)
        def_policy = JPEGFilterDefense(policy)

    :param policy: (policy) Tianshou Policy object
    :param quality: (int) jpeg image quality
    :return: defended policy
    """

    f = policy.forward

    def defense_forward(s, last_state=None):
        """
        :param s: Tianshou Batch object
        :return: defended policy.forward method
        """

        x = s.obs
        for batch in range(x.shape[0]):
            for frame in range(x.shape[1]):
                img = _to_pil_image(np.expand_dims(x[batch][frame].astype(np.uint8), axis=-1))
                virtualpath = BytesIO()
                img.save(virtualpath, 'JPEG', quality=quality)
                arr = np.array(Image.open(virtualpath))
                img = Image.fromarray(np.uint8(arr))
                virtualpath = BytesIO()
                img.save(virtualpath, 'BMP')
                x[batch][frame] = np.array(Image.open(virtualpath))
        s.obs = x
        return f(s, last_state)

    policy.forward = defense_forward
    return policy
