import os
import random
import torch.nn as nn
import numpy as np
import torch
import torch.nn.parallel
import pickle as pkl
import torch.backends.cudnn as cudnn
from utils import get_net_builder, get_logger, get_port, over_write_args_from_file, ramps, AverageMeter, init_params
from data.get_dataset import get_discover_datasets
from tqdm import tqdm
from argparse import ArgumentParser
from tqdm import tqdm
from utils.utils import model_statistics
import ipdb
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment
from functools import partial
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    dataset = get_discover_datasets(args.dataset, args)
    train_unlabel_dataset, train_label_dataset = dataset['val_unlabel_dataset'], dataset['val_label_dataset']

    train_unlabel_dataloader = torch.utils.data.DataLoader(
            train_unlabel_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    train_label_dataloader = torch.utils.data.DataLoader(
            train_label_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    _net_builder = get_net_builder(args.net, args.net_from_name)

    model = _net_builder(num_classes=100, num_unseen_classes=args.num_unlabeled_classes,
                         num_seen_classes=args.num_labeled_classes, pretrained=args.use_pretrain,
                         pretrained_path=args.pretrain_path)


    for m in model.parameters():
        m.requires_grad = False
    model_statistics(model)
    model = model.to(args.device)

    print('Testing on all in the training data...')

    binary_search(model, merge_test_loader=[train_unlabel_dataloader, train_label_dataloader], args=args)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((int(D), int(D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(int(y_pred[i])), int(y_true[i])] += 1
    mapping = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    acc = sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size
    return acc


def test_kmeans_for_scipy(K, merge_test_loader, model, args=None, verbose=False):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    K = int(K)

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
    prefix = f"{args.dataset}_{args.lab_imbalance_factor}_{args.imbalance_factor}_{args.num_labeled_classes}_{args.num_unlabeled_classes}"
    pkl_path = os.path.join(f"{prefix}_feature.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            all_feats = pkl.load(f)
        pkl_path = os.path.join(f"{prefix}_mask.pkl")
        with open(pkl_path, "rb") as f:
            mask_lab = pkl.load(f)
        pkl_path = os.path.join(f"{prefix}_target.pkl")
        with open(pkl_path, "rb") as f:
            targets = pkl.load(f)
    else:
        # First extract all features
        model.eval()
        for loader in merge_test_loader:
            for id, batch in enumerate(tqdm(loader)):
                images, labels, _ = batch

                images = images.to(args.device)
                labels = labels.to(args.device)
                mask_lb = labels < args.num_labeled_classes

                with torch.no_grad():
                    outputs = model(images)
                feats = outputs['feat']

                feats = torch.nn.functional.normalize(feats, dim=-1)

                all_feats.append(feats.cpu().detach().numpy())
                targets = np.append(targets, labels.cpu().numpy())
                mask_lab = np.append(mask_lab, mask_lb.cpu().bool().numpy())
        mask_lab = mask_lab.astype(bool)

        all_feats = np.concatenate(all_feats)

        with open(os.path.join(f"{prefix}_feature.pkl"), "wb") as f:
            pkl.dump(all_feats, f)
        with open(os.path.join(f"{prefix}_mask.pkl"), "wb") as f:
            pkl.dump(mask_lab, f)
        with open(os.path.join(f"{prefix}_target.pkl"), "wb") as f:
            pkl.dump(targets, f)


    print(f'Fitting K-Means for K = {K}...')


    clustering = AgglomerativeClustering(n_clusters=K).fit(all_feats)
    preds = clustering.labels_


    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask_lab

    sample_labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                                      nmi_score(targets[mask], preds[mask]), \
                                                      ari_score(targets[mask], preds[mask])
    class_labelled_acc, labelled_nmi, labelled_ari = class_cluster_acc(targets.astype(int)[mask],
                                                                       preds.astype(int)[mask]), \
                                                     nmi_score(targets[mask], preds[mask]), \
                                                     ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = class_cluster_acc(targets.astype(int)[~mask],
                                                                       preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])
    #acc = sample_labelled_acc
    print(f'K = {K}')

    print('Unlabelled class wise acc: ')
    print(unlabelled_acc)
    print('Labelled class wise acc: ')
    print(class_labelled_acc)
    # acc = sample_labelled_acc
    acc = sample_labelled_acc*args.alpha + class_labelled_acc*(1 - args.alpha)
    # acc = class_labelled_acc
    print('Labelled Instances acc: ')
    print(acc)

    return acc

def compute_best_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    www = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        www[y_pred[i], y_true[i]] += 1
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w))), w.T, www.T

def binary_search(model, merge_test_loader, args):
    from time import time
    start = time()
    min_classes = args.num_labeled_classes
    # Iter 0
    big_k = args.max_classes
    small_k = min_classes
    diff = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    labelled_acc_big = test_kmeans_for_scipy(big_k, merge_test_loader, model, args)
    labelled_acc_small = test_kmeans_for_scipy(small_k, merge_test_loader, model, args)
    labelled_acc_middle = test_kmeans_for_scipy(middle_k, merge_test_loader, model, args)

    print(f'Iter 0: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
    all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    for i in range(1, int(np.log2(diff)) + 1):

        if labelled_acc_big > labelled_acc_small:

            best_acc = max(labelled_acc_middle, labelled_acc_big)

            small_k = middle_k
            labelled_acc_small = labelled_acc_middle
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)

        else:

            best_acc = max(labelled_acc_middle, labelled_acc_small)
            big_k = middle_k

            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            labelled_acc_big = labelled_acc_middle

        labelled_acc_middle = test_kmeans_for_scipy(middle_k, merge_test_loader, model, args)

        print(f'Iter {i}: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
        all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
        print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')
    print("The time cost is: {}".format(time() - start))
    return best_acc_at_k, best_acc_so_far


def class_cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((int(D), int(D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(int(y_pred[i])), int(y_true[i])] += 1
    mapping = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    s = sum(w)
    acc = sum([w[i, j] / s[j] if s[j]!=0  else 0 for i, j in mapping]) * 1.0 / (y_true.max() - y_true.min())
    return acc


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='saved_models/UNO')
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
    parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
    parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--alpha", default=0.5, type=float, help="weight")

    parser.add_argument("--lr", default=0.4, type=float, help="learning rate")
    parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
    parser.add_argument('--amp', default=False, action="store_true", help='use mixed precision training or not')
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="warmup epochs")
    parser.add_argument("--max_epochs", default=50, type=int, help="warmup epochs")
    parser.add_argument('--net', type=str, default='vit_small_patch2_32')
    parser.add_argument('--net_from_name', action="store_true", default=False)
    parser.add_argument('--use_pretrain', default=True, action='store_true')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')

    parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
    parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
    parser.add_argument("--overcluster_factor", default=1, type=int, help="overclustering factor")
    parser.add_argument("--num_heads", default=5, type=int, help="number of heads for clustering")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.5, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--temperature", default=1, type=float, help="softmax temperature")
    parser.add_argument("--project", default="MIv2", type=str, help="wandb project")
    parser.add_argument("--entity", default="rikkixu", type=str, help="wandb entity")
    parser.add_argument("--offline", default=True, action="store_true", help="disable wandb")
    parser.add_argument("--num_labeled_classes", default=50, type=int, help="number of labeled classes")
    parser.add_argument("--num_unlabeled_classes", default=50, type=int, help="number of unlab classes")
    parser.add_argument("--pretrain_path", type=str, help="pretrained checkpoint path")
    parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
    parser.add_argument("--ss_pretrained", default=False, action="store_true", help="self-supervised pretrain")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu")
    parser.add_argument("--ratio", type=float, default=50, help="the percentage of labeled data")
    parser.add_argument("--regenerate", default=False, action="store_true", help="whether to generate data again")
    parser.add_argument("--resume", default=False, action="store_true", help="whether to use old model")
    parser.add_argument("--save-model", default=False, action="store_true", help="whether to save model")
    parser.add_argument('-sn', '--save_name', type=str)
    parser.add_argument('--min_max_ratio', type=float, default=0.3)
    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--k_means_init', type=int, default=10)
    parser.add_argument("--algorithm",
                        default='UNO',
                        type=str,
                        help="")
    parser.add_argument("--lab_imbalance_factor",
                        type=int,
                        help="")
    parser.add_argument("--imbalance_factor",
                        type=int,
                        help="")
    parser.add_argument("--num_classes",
                        default=100,
                        type=int,
                        help="number of small crops")
    parser.add_argument("--random_head",
                        default=False,
                        action="store_true",
                        help="")

    parser.add_argument("--lr_w",
                        type=float,
                        help="")

    parser.add_argument("--gamma", type=float)
    parser.add_argument("--num_outer_iters",
                        type=int,
                        help="")
    parser.add_argument("--flag",
                        type=int,
                        default=0,
                        help="")

    parser.add_argument("--val",
                        type=bool,
                        default=False,
                        help="")
    parser.add_argument('--max_classes', default=1000, type=int)


    # config file
    parser.add_argument('--c', type=str, default='')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    over_write_args_from_file(args, args.c)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.num_crops = args.num_large_crops
    return args


if __name__ == "__main__":
    args = get_config()
    device = torch.device("cuda" if args.cuda else "cpu")

    args.dump_path = os.path.join(
        "data/splits", f'{args.dataset}-labeled-{args.num_labeled_classes}'
        f'-unlabeled-{args.num_unlabeled_classes}-'
        f'imbalance-{args.imbalance_factor}.pkl')

    os.environ["WANDB_API_KEY"] = "4da1b870fbd955fdee5d0ebb0f28e94ebdaae96d"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.val = False
    save_path = os.path.join(args.save_dir, args.save_name)
    print(save_path)
    args.save_path = save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(args)
    main(args)