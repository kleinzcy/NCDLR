import argparse
import os
import random as randd
import warnings
import torch.nn as nn
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from utils import get_net_builder, get_logger, get_port, over_write_args_from_file, AverageMeter, init_params, \
    model_statistics
from data.get_dataset import get_discover_datasets

from utils.utils import split_cluster_acc_v2, split_cluster_acc_v2_class
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns


def get_config():
    parser = argparse.ArgumentParser(description='NCD')

    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_labeled_classes', type=int, default=50)
    parser.add_argument('--num_unlabeled_classes', type=int, default=50)
    parser.add_argument("--regenerate", default=False, action="store_true", help="whether to generate data again")
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument("--imbalance-factor", default=1, type=int, help="imbalance factor")

    parser.add_argument('-bsz', '--batch_size', type=int, default=128)

    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='layer-wise learning rate decay, default to 1.0 which means no layer decay')

    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', action="store_true", default=False)
    parser.add_argument('--use_pretrain', default=False, action='store_true')
    parser.add_argument('--pretrain_path', default='', type=str)
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    ## standard setting configurations
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('-nc', '--num_classes', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=10)

    ## cv dataset arguments
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=float, default=0.875)

    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument("--project", default="MIv2", type=str, help="wandb project")
    parser.add_argument("--entity", default="rikkixu", type=str, help="wandb entity")
    parser.add_argument("--offline", default=True, action="store_true", help="disable wandb")

    parser.add_argument('--knn', default=-1, type=int)
    parser.add_argument('--m_size', default=2000, type=int)
    parser.add_argument('--m_t', type=float, default=0.05)
    parser.add_argument('--w_pos', type=float, default=0.2)
    parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
    parser.add_argument('--increment_coefficient', type=float, default=0.05)

    # config file
    parser.add_argument('--c', type=str, default='')

    # add algorithm specific parameters
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    over_write_args_from_file(args, args.c)

    args.num_crops = args.num_large_crops
    return args


def main(args):
    save_path = os.path.join(args.save_dir, args.save_name)

    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu == 'None':
        args.gpu = None
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    '''
    main_worker is conducted on each GPU.
    '''

    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    randd.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)

    model = _net_builder(num_classes=100, num_unseen_classes=args.num_unlabeled_classes,
                         num_seen_classes=args.num_labeled_classes, pretrained=args.use_pretrain,
                         pretrained_path=args.pretrain_path)

    model = init_params(model, random_head=args.random_head)
    model_statistics(model)
    model = model.cuda()

    checkpoint = torch.load('path')

    model.load_state_dict(checkpoint)
    datasets = get_discover_datasets(args.dataset, args)
    train_dataset, labeled_train_dataset, unlabeled_train_dataset, all_eval_dataset = datasets["train_dataset"], \
                                                                                      datasets["train_label_dataset"], \
                                                                                      datasets['train_unlabel_dataset'], \
                                                                                      datasets['test_dataset']
    class_unlabel_nums = datasets["class_unlabel_nums"]

    all_eval_loader = torch.utils.data.DataLoader(
        all_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    state = {k: v for k, v in args._get_kwargs()}
    wandb.init(project=args.project, entity=args.entity, config=state, name=args.save_name, dir="logs")


    args.head = 'test'
    test_results = test(model, all_eval_loader, args, class_unlabel_nums, 'test')

    # train(model, train_loader, unlabeled_eval_loader, all_eval_loader, class_unlabel_nums, args, wandb)
    print("Test-All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]--"
          "Head-[{:.2f}]--Medium-[{:.2f}]--Tail-[{:.2f}]"
          .format(test_results["test/all/avg"] * 100, test_results["test/novel/avg"] * 100,
                  test_results["test/seen/avg"] * 100, test_results["test/head/avg"] * 100,
                  test_results["test/medium/avg"] * 100, test_results["test/tail/avg"] * 100))


def test(model, test_loader, args, class_unlabel_nums, prefix):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        outputs = model(x)

        if args.head == 'head1':
            output = outputs['seen_logits']
        elif args.head == 'head2':
            output = outputs['unseen_logits']
        else:
            output = torch.cat([outputs['seen_logits'], outputs['unseen_logits']], dim=-1)
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
    results = {}
    if args.dataset == 'herbarium_19':
        _res, _ = split_cluster_acc_v2_class(args, targets.astype(int), preds.astype(int),
                                       class_unlabel_nums=class_unlabel_nums,
                                       num_seen=args.num_labeled_classes)
    else:

        _res,_ = split_cluster_acc_v2(args, targets.astype(int), preds.astype(int), class_unlabel_nums=class_unlabel_nums,
                                num_seen=args.num_labeled_classes)
    for key, value in _res.items():
        if key in results.keys():
            results[key].append(value)
        else:
            results[key] = [value]
    log = {}
    for key, value in results.items():
        log[prefix + "/" + key + "/" + "avg"] = round(sum(value) / len(value), 4)
    return log


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


if __name__ == "__main__":
    args = get_config()
    args.comment = args.save_name
    device = torch.device("cuda" if args.cuda else "cpu")
    os.environ["WANDB_API_KEY"] = "f671d0a982a7253de2fa46fad5249cb94fd943c9"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"
    port = get_port()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.val = False
    main(args)
