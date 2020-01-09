import random
import torch
import numpy as np
import torch.nn as nn

SMOOTHER = 0.999999


def get_random_size(imsize, w, h):
    scale = w / h
    DELTA = 8
    L = imsize[0]
    H = imsize[1]
    if scale >= 1:
        nw = random.randint(L, H)
        nh = int(nw / scale)
        nh += random.randint(-DELTA, DELTA)
    else:
        nh = random.randint(L, H)
        nw = int(nh * scale)
        nw += random.randint(-DELTA, DELTA)
    return (nw, nh)


def do_whiten(b, m, p):
    b = p @ (b - m)
    b = b / (b.norm(dim=0) + 1e-6)
    return b


def one_hot(size, index):
    mask = torch.LongTensor(index.size(0), size).fill_(0).to(index.device)
    ret = mask.scatter_(1, index.view(-1, 1), 1)
    return ret


# def w2img(w, eps):
#     return w
#     return torch.tanh((w)/SMOOTHER) * eps


def rescale_check(check, sat, sat_change, sat_min):
    return sat_change < check and sat > sat_min


def inv_gfr(attack, baseline):
    s = 0
    for k in attack.keys():
        s += sum(
            [
                abs(attack[k][i] - baseline[k][i]) / baseline[k][i]
                for i in range(len(attack[k]))
            ]
        )
    return 1 - s / 14


class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        self.weight = nn.Parameter(torch.zeros(3))

    def forward(self, losses):
        w = (-self.weight).exp()
        return torch.dot(w, losses) + self.weight.sum()


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    def str(s, color):
        return color + s + bcolors.ENDC


def idcg(n_rel):
    """idcg

    :n_rel: number of real doc
    :returns: value of idcg

    """
    nums = np.ones(n_rel)
    denoms = np.log2(np.arange(n_rel) + 2)
    return (nums / denoms).sum()
