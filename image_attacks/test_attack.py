from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from image_attacks.FGSM import fast_gradient_method
from image_attacks.train_CNN import CNN


class SimpleModel(torch.nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.w1 = torch.tensor([[1.5, .3], [-2, .3]])
        self.w2 = torch.tensor([[-2.4, 1.2], [.5, -2.3]])

    def forward(self, x):
        x = torch.matmul(x, self.w1)
        x = torch.sigmoid(x)
        x = torch.matmul(x, self.w2)
        return x


if __name__ == '__main__':
    model = SimpleModel()
    x = torch.randn(100, 2)  # samples from N(0,1)
    normalized_x = torch.rand(100, 2)  # samples from U(0,1)
    red_ind = list(range(1, len(x.size())))
    norm_list = [1, 2, np.inf]
    eps_list = [0.0, 0.1, 0.2, 0.3]

    print("Untargeted")
    for norm in norm_list:
        for eps in eps_list:
            x_adv = fast_gradient_method(model, normalized_x, eps, norm, targeted=False)
            _, ori_label = model(normalized_x).max(1)
            _, adv_label = model(x_adv).max(1)
            adv_acc = np.array((adv_label.eq(ori_label).sum().to(torch.float) / normalized_x.size(0)))
            print("Norm: {}, eps: {}, acc: {:.2f}".format(norm, eps, adv_acc))
