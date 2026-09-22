import torch
from torch import nn


class RunningMeanStd(nn.Module):
    def __init__(self, shape):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("mean", torch.zeros((1,) + shape))
        self.register_buffer("S", torch.ones((1,) + shape) * 1e-4)
        self.register_buffer("std", torch.sqrt(self.S))

    def update(self, x):
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        self.n += 1
        old_mean = self.mean.clone()
        new_mean = x.mean(dim=0, keepdim=True)
        self.mean = old_mean + (new_mean - old_mean) / self.n
        self.S = (
            self.S
            + (x - new_mean).pow(2).mean(dim=0, keepdim=True)
            + (old_mean - new_mean).pow(2) * (self.n - 1) / self.n
        )
        self.std = torch.sqrt(self.S / self.n)
        return


class Normalization(nn.Module):
    def __init__(self, shape, max_update=2000):
        super().__init__()
        self.running_ms = RunningMeanStd(shape=shape)
        self.register_buffer("max_update", torch.tensor(max_update))

    def __call__(self, x, update=True):
        if self.training and self.max_update > 0 and update:
            self.running_ms.update(x)
            self.max_update -= 1
        x = (x - self.running_ms.mean) / self.running_ms.std
        return x

    def inv(self, x):
        x = x * self.running_ms.std + self.running_ms.mean
        return x
