import torch
import torch.nn as nn
import math

# https://github.com/usuyama/Humpback-Whale-Identification-Challenge-2019_2nd_palce_solution/blob/master/net/MagrinLinear.py


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class MarginLinear(nn.Module):
    # ArcFace: cos(th + m)
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=10008, s=64., m=0.5):
        super(MarginLinear, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel (per-class embeddings)
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label, is_infer=False):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)

        # calculate cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        # cos(x + m) = cos(x) * cos(m) - sin(x) * sin(m)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use CosFace instead cos(th) - m
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta

        if not is_infer:
            # swap the value of the target-label to the one with margin
            idx_ = torch.arange(0, nB, dtype=torch.long)
            output[idx_, label] = cos_theta_m[idx_, label]

        output *= self.s  # scale up in order to make softmax work, first introduced in normface

        return output


if __name__ == '__main__':
    import numpy as np

    emb_size = 3
    num_classes = 5

    labels = torch.from_numpy(np.array([0] * 7 + [1] * 4))
    embeddings = torch.randn(len(labels), emb_size)

    margin_linear = MarginLinear(embedding_size=emb_size, classnum=num_classes, s=1.0)
    print(margin_linear(embeddings, labels, is_infer=False))
    print("*" * 20)
    print(margin_linear(embeddings, labels, is_infer=True))
    print("*" * 40)

    margin_linear = MarginLinear(embedding_size=emb_size, classnum=num_classes)
    print(margin_linear(embeddings, labels, is_infer=False))
    print("*" * 20)
    print(margin_linear(embeddings, labels, is_infer=True))
