# coding=utf-8
"""
Tau-normalized classifier for long-tailed data
"""
import torch
import numpy as np


def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws


def dotproduct_similarity(A, B):
    AB = torch.mm(A, B.t())
    return AB


def logits2preds(logits):
    _, nns = logits.max(dim=1)
    preds = np.array([i for i in nns])
    return preds


def forward(weights, features, batch_size):
    total_logits = []
    test_size = features.shape[0]
    for i in range(test_size // batch_size + 1):
        feat = features[i * batch_size: (i + 1) * batch_size]
        feat = torch.Tensor(feat)
        logits = dotproduct_similarity(feat, weights)
        total_logits.append(logits)
    total_logits = torch.cat(total_logits)
    return total_logits


def tau_norm_classifier(weights, testing_set, batch_size):
    all_preds = []
    for p in np.linspace(0, 2, 21):
        ws = pnorm(weights, p)
        logits = forward(ws, testing_set, batch_size)
        preds = logits2preds(logits)
        all_preds.append(preds)
    return all_preds




