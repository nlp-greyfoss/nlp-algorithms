import math

import torch


def swish(x):
    return x * torch.sigmoid(x)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
