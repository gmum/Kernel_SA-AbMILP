from __future__ import print_function
import torch
import torch.utils.data

min_epsilon = 1e-5
max_epsilon = 1.-1e-5


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x , 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Bernoulli(x, mean, average=False, dim=1):
    probs = torch.clamp(mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if average:
        return torch.mean(log_bernoulli, dim)
    else:
        return torch.sum(log_bernoulli, dim)


def log_Bernoulli_DS(x, mean, average=False, dim=1):
    probs = torch.clamp(mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    return log_bernoulli
