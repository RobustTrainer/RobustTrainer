# coding=utf-8
from scipy.special import softmax
import faiss
import torch
import numpy as np
import torch.nn as nn


def get_prototypes(x, y, args):
    cluster = []
    if args.type_prototypes == "KMEANS":
        for i in range(args.nb_classes):
            y_x = x[y == i]
            if len(y_x) < args.nb_prototypes:
                cluster.append(y_x)
            else:
                cluster.append(kmeans(y_x, args))
        return cluster
    elif args.type_prototypes == "TopK":
        for i in range(args.nb_classes):
            y_x = x[y == i]
            cluster.append(topk(y_x, args))
        return cluster
    else:
        raise ValueError("Invalid prototype types")


def kmeans(x, args):
    d = x.shape[1]
    k = int(args.nb_prototypes)
    cluster = faiss.Clustering(d, k)
    cluster.verbose = True
    cluster.niter = 20
    cluster.nredo = 5

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    index = faiss.GpuIndexFlatIP(res, d, cfg)

    cluster.train(x, index)
    centroids = faiss.vector_to_array(cluster.centroids).reshape(k, d)

    return centroids


def cosine_similarity(x1, x2):
    return x1.dot(x2.T)


def calculate_rou(S, rate=0.4):
    m = S.shape[0]

    t = int(rate * m * m)
    temp = np.sort(S.reshape((m*m, )))
    Sc = temp[-t]

    rou = np.sum(np.sign(S-Sc), axis=1) - np.sign(S.diagonal() - Sc)
    return rou


def get_prototypes_topk_index(S, rou, p):
    rou_max = np.max(rou)
    m = S.shape[0]
    ita = np.zeros(m)

    for i in range(m):
        if rou[i] == rou_max:
            ita[i] = np.min(S[i])
        else:
            ita[i] = S[i, i]
            for j in range(m):
                if i != j and rou[j] > rou[i]:
                    if ita[i] < S[i, j]:
                        ita[i] = S[i, j]
    return np.argsort(ita)[:p]


def topk(x, args):
    if len(x) < args.nb_rand_samples:
        args.nb_rand_samples = len(x)
    z_samples_indexes = np.random.choice(np.arange(x.shape[0]), args.nb_rand_samples, replace=False)
    z_samples = x[z_samples_indexes]
    S = cosine_similarity(z_samples, z_samples)
    rou = calculate_rou(S)
    prototype_indexes = get_prototypes_topk_index(S, rou, args.nb_prototypes)
    prototypes = z_samples[prototype_indexes]
    return prototypes


def label_clean(x, y, args):
    prototypes = get_prototypes(x, y, args)
    prototypes = np.vstack(prototypes)
    logits_proto = x.dot(prototypes.T) / args.temperature
    softmax_proto = softmax(logits_proto, axis=1)
    soft_proto = np.zeros((x.shape[0], args.nb_classes), dtype=np.float64)
    for i in range(args.nb_classes):
        soft_proto[:, i] = np.sum(softmax_proto[:, i * args.nb_prototypes:(i+1) * args.nb_prototypes], axis=1)

    gt_score = soft_proto[np.arange(x.shape[0]), y]
    id_idx = gt_score > args.ood_threshold

    max_score = np.max(soft_proto, axis=1)
    hard_proto = np.argmax(soft_proto, axis=1)
    clean_idx = (hard_proto == y) & id_idx
    noisy_idx = (hard_proto != y) & id_idx
    pseudo4noise = (max_score > args.pseudo_threshold) & noisy_idx

    y[pseudo4noise] = hard_proto[pseudo4noise]

    selected_x = x[id_idx]
    selected_y = y[id_idx]

    return selected_x, selected_y, clean_idx, id_idx

