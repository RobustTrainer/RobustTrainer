# coding=utf-8
"""
A set of util functions for model training
"""
import torch
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def accuracy(outputs, targets, topk=(1,)):
    # compute the precision@k
    max_k = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.avg = round(self.sum / self.count, 4)


class DAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}

    def update(self, values):
        assert isinstance(values, dict)
        for key, val in values.items():
            if not (key in self.values):
                self.values[key] = AverageMeter()
            self.values[key].update(val)

    def average(self):
        average = {}
        for key, val in self.values.items():
            average[key] = val.avg
        return average

    def __str__(self):
        ave_stats = self.average()
        return ave_stats.__str__()