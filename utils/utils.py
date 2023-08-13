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
    head_acc = 0
    med_acc = 0
    tai_acc = 0

    return {"all": total_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": old_acc,
            "novel": new_acc, "novel_nmi": novel_nmi, "novel_ari": novel_ari,
            'head': head_acc, 'medium': med_acc, 'tail': tai_acc}, mapping

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

    if args.dataset in ['ImageNet100', "Place365"]:
        head_acc = torch.mean(torch.tensor(res)[:15])
        med_acc = torch.mean(torch.tensor(res)[15:35])
        tai_acc = torch.mean(torch.tensor(res)[35:])
    elif args.dataset in ["herbarium_19", "inaturalist18"]:
        head_acc = torch.mean(torch.tensor(res)[torch.tensor(class_unlabel_nums['head'])-num_seen])
        med_acc = torch.mean(torch.tensor(res)[torch.tensor(class_unlabel_nums['med'])-num_seen])
        tai_acc = torch.mean(torch.tensor(res)[torch.tensor(class_unlabel_nums['tail'])-num_seen])

    else:
        head_acc = torch.mean(torch.tensor(res)[(class_unlabel_nums[50:] >= 100).tolist()])
        tai_acc = torch.mean(torch.tensor(res)[(class_unlabel_nums[50:] < 20).tolist()])
        med_acc = torch.mean(
            torch.tensor(res)[((class_unlabel_nums[50:] < 100) & (class_unlabel_nums[50:] >= 20)).tolist()])


    return {"all": total_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": old_acc,
            "novel": new_acc, "novel_nmi": novel_nmi, "novel_ari": novel_ari,
            'head': head_acc.item(), 'medium': med_acc.item(), 'tail': tai_acc.item()}, mapping

def split_cluster_acc_v1(args, y_true, y_pred, class_unlabel_nums, num_seen):
    global best_unseen_acc
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
    weight = mask.mean()

    seen_acc = np.mean(y_true[mask] == y_pred[mask])

    novel_acc, res, w1, map1 = cluster_acc(args, y_true[~mask], y_pred[~mask])
    best_unseen_acc = max(novel_acc, best_unseen_acc)
    novel_nmi = metrics.normalized_mutual_info_score(y_true[~mask], y_pred[~mask])
    novel_ari = metrics.adjusted_rand_score(y_true[~mask], y_pred[~mask])

    all_acc, _, w2, map2 = cluster_acc(args, y_true, y_pred, all=True)
    all_nmi = metrics.normalized_mutual_info_score(y_true[mask], y_pred[mask])
    all_ari = metrics.adjusted_rand_score(y_true[mask], y_pred[mask])

    if args.dataset == 'ImageNet100':
        head_acc = torch.mean(torch.tensor(res)[:15])
        med_acc = torch.mean(torch.tensor(res)[15:35])
        tai_acc = torch.mean(torch.tensor(res)[35:])

    else:

        head_acc = torch.mean(torch.tensor(res)[(class_unlabel_nums[50:] >= 100).tolist()])
        tai_acc = torch.mean(torch.tensor(res)[(class_unlabel_nums[50:] < 20).tolist()])
        med_acc = torch.mean(
            torch.tensor(res)[((class_unlabel_nums[50:] < 100) & (class_unlabel_nums[50:] >= 20)).tolist()])



    if best_unseen_acc == novel_acc and class_unlabel_nums != None:
        mask1 = np.ones_like(w1)
        for i, j in map1:
            mask1[j, i] = 0
        mask1 = ~mask1.astype(np.bool)
        plt.figure(figsize=(200, 150))
        plt.title(args.save_name + "_unlabel_hm")
        x1 = sns.heatmap(w1.T, cmap="YlGnBu", mask=mask1, annot=True, cbar=False)
        # sns.heatmap(w1.T,mask=~mask1,cmap="autumn", annot=True,cbar=False)
        fig1 = x1.get_figure()
        fig1.savefig(os.path.join(args.save_path, "unlabel_hm.png"))

        plt.close()
        mask2 = np.ones_like(w2)
        for i, j in map2:
            mask2[i, j] = 0
        mask2 = ~mask2.astype(np.bool)
        plt.figure(figsize=(200, 150))
        plt.title(args.save_name + "_all_hm")
        x2 = sns.heatmap(w2.T, cmap="YlGnBu", mask=mask2, annot=True, cbar=False)
        # sns.heatmap(w2.T, mask=~mask2, cmap="autumn", annot=True, cbar=False)

        fig2 = x2.get_figure()
        fig2.savefig(os.path.join(args.save_path, "all_hm.png"))
        plt.close()
        X = np.array(range(50))
        Y = res
        plt.bar(X, Y, 0.4, color="orange")
        plt.title(args.save_name + '_' + str(best_unseen_acc))
        plt.savefig(os.path.join(args.save_path, args.save_name + ".jpg"))
        plt.close()


    return {"all": all_acc, "all_nmi": all_nmi, "all_ari": all_ari, "seen": seen_acc,
            "novel": novel_acc, "novel_nmi": novel_nmi, "novel_ari": novel_ari,
            'head': head_acc.item(), 'medium': med_acc.item(), 'tail': tai_acc.item()}


def cluster_acc(args, y_true, y_pred, all=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mapping, w = compute_best_mapping(y_true, y_pred)
    ww = w
    detail_all = []

    if not all:
        for kk, qq in mapping:
            if qq >= 50:
                detail_all.append((qq, float(ww[kk, qq] / np.sum(y_true == qq))))
        detail_all = sorted(detail_all)
    res = [i for j, i in detail_all]
    return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size, res, ww, mapping


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


class BCE(nn.Module):
    eps = 1e-7

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),
                                                                                            str(len(prob2)),
                                                                                            str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        # dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


def roc(similarity, ground_truth, epoch, save_dir):
    """
    similarity: 1D array, each element represent the similarity of two samples.
    ground_truth: 1D array, pair-wise ground truth, each element represent the ground truth of pair wise similarity.
    """
    plt.figure()
    precision, recall, thresholds = precision_recall_curve(ground_truth, similarity)
    plt.plot(thresholds, precision[:-1], label="precision")
    plt.plot(thresholds, recall[:-1], label="recall")
    plt.axvline(0.95)
    plt.legend()
    plt.grid()
    x_major_locator = MultipleLocator(0.05)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.title(f"Epoch {epoch}")
    plt.savefig(os.path.join(save_dir, f"roc_{epoch}.pdf"), dpi=300, bbox_inches='tight', pad_inches = 0)
    plt.close()

