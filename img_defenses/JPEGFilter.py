from torchvision import transforms
from PIL import Image

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
    """

    f = policy.forward

    def defense_forward(s, last_state=None):
        """
        :param s: Tianshou Batch object
        :return: defended policy.forward method
        """

        x = s.obs
        for batch in range(len(x)):
            for frame in range(batch):
                img = _to_pil_image(x[batch][frame])
                virtualpath = BytesIO()
                img.save(virtualpath, 'JPEG', quality=quality)
                x[batch][frame] = Image.open(virtualpath)
        s.obs = x
        return f(s, last_state)

    policy.forward = defense_forward
    return policy
