import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np

# https://kyudy.hatenablog.com/entry/2019/05/20/105526


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    print(index)
    return mask.scatter_(1, index, ones)


# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class FocalLossWithOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()


class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss

        return loss.mean()


if __name__ == "__main__":
    device = torch.device("cuda")

    focal_without_onehot = FocalLossWithOutOneHot(gamma=1)
    focal_with_onehot = FocalLossWithOneHot(gamma=1)
    input = torch.Tensor([[0.3, 0.1, 0.1], [0.3, 0.6, 0.001], [0.01, 0.002, 2.3], [0.01, 0.002, 2.3]]).to(device)
    target = torch.Tensor([0, 1, 1, 2]).long().to(device)

    print(input, target)

    print(focal_without_onehot(input, target))
    # exception will occur when input and target are stored to GPU(s).
    # focal_with_onehot(input, target)
    print(FocalLossWithOutOneHot(gamma=0)(input, target))
    print(F.cross_entropy(input, target, reduction='mean'))
