# _*_ coding: utf-8 _*_
"""
Time:     2022/9/26 12:17
Author:   Ruijie Xu
File:     utils.py
"""
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import numpy as np
import torch.nn as nn

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle as pkl
import torch
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from matplotlib.pyplot import MultipleLocator


best_unseen_acc = 0
def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def model_statistics(model):
    total_params = sum(param.numel() for param in model.parameters()) / 1000000.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    print('    Total params: %.2fM, trainable params: %.2fM' % (total_params, trainable_params))


def init_params(model, grad_from_block=11, random_head=False):
    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in model.parameters():
        m.requires_grad = False
    # Only finetune layers from block 'args.grad_from_block' onwards
    if random_head is True:
        print("Warning: You are fixing the head parameters!!!")
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= grad_from_block:
                m.requires_grad = True
        elif "head" in name:
            if not random_head:
                m.requires_grad = True
    return model


def init_cls(model):
    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in model.parameters():
        m.requires_grad = False
    # Only finetune cls
    for name, m in model.named_parameters():
        if "head" in name:
            m.requires_grad = True
            if len(m.shape) < 2:
               nn.init.kaiming_uniform_(m.unsqueeze(0))
            else:
               nn.init.kaiming_uniform_(m)
            print(name)
    return model
    

def split_cluster_acc_v2_class(args, y_true, y_pred, class_unlabel_nums, num_seen):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets
    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]`
    """
    if args.est_k:
        num_unlabeled_classes = args.num_unlabeled_classes - args.error
    else:
        num_unlabeled_classes = args.num_unlabeled_classes
    mask = y_true < num_seen
    y_true = y_true.astype(int)

    novel_nmi = metrics.normalized_mutual_info_score(y_true[~mask], y_pred[~mask])
    novel_ari = metrics.adjusted_rand_score(y_true[~mask], y_pred[~mask])
    all_nmi = metrics.normalized_mutual_info_score(y_true[mask], y_pred[mask])
    all_ari = metrics.adjusted_rand_score(y_true[mask], y_pred[mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    mapping = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))

    sum_w = sum(w)
    total_acc = sum([w[i, j] / sum_w[j] if sum_w[j]!= 0 else 0 for i, j in mapping]) * 1.0 / (args.num_labeled_classes + num_unlabeled_classes)
    old_acc = sum([w[i, j] / sum_w[j] if sum_w[j] != 0 and j < args.num_labeled_classes else 0 for i, j in mapping]) * 1.0 / args.num_labeled_classes
    new_acc = sum([w[i, j] / sum_w[j] if sum_w[j] != 0 and j >= args.num_labeled_classes else 0 for i, j in mapping]) * 1.0 / num_unlabeled_classes

    return {"all": total_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": old_acc,
            "novel": new_acc, "novel_nmi": novel_nmi, "novel_ari": novel_ari}, mapping

def split_cluster_acc_v2(args, y_true, y_pred, class_unlabel_nums, num_seen, only_unseen=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets
    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """

    mask = y_true < num_seen
    y_true = y_true.astype(int)
    total_class = y_true.max()
    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    novel_nmi = metrics.normalized_mutual_info_score(y_true[~mask], y_pred[~mask])
    novel_ari = metrics.adjusted_rand_score(y_true[~mask], y_pred[~mask])
    all_nmi = metrics.normalized_mutual_info_score(y_true[mask], y_pred[mask])
    all_ari = metrics.adjusted_rand_score(y_true[mask], y_pred[mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}

    mapping = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    ww = w
    detail_all = []

    for kk, qq in mapping:
        if qq >= num_seen and qq < total_class + 1:
            detail_all.append((qq, float(ww[kk, qq] / np.sum(y_true == qq))))
        detail_all = sorted(detail_all)
    res = [i for j, i in detail_all]


    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])


    if only_unseen:
        old_acc = 0
    else:
        old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return {"all": total_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": old_acc,
            "novel": new_acc, "novel_nmi": novel_nmi, "novel_ari": novel_ari}, mapping

def split_cluster_acc_v1(args, y_true, y_pred, class_unlabel_nums, num_seen):
    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = y_true < num_seen
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    seen_acc = np.mean(y_true[mask] == y_pred[mask])

    novel_acc, _ = cluster_acc(args, y_true[~mask], y_pred[~mask])
    novel_nmi = metrics.normalized_mutual_info_score(y_true[~mask], y_pred[~mask])
    novel_ari = metrics.adjusted_rand_score(y_true[~mask], y_pred[~mask])

    all_acc, _ = cluster_acc(args, y_true, y_pred, all=True)
    all_nmi = metrics.normalized_mutual_info_score(y_true[mask], y_pred[mask])
    all_ari = metrics.adjusted_rand_score(y_true[mask], y_pred[mask])


    return {"all": all_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": seen_acc,
            "novel": novel_acc, "novel_nmi": novel_nmi, "novel_ari": novel_ari}


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mapping, w = compute_best_mapping(y_true, y_pred)
    return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size, mapping


def compute_best_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w))), w


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

