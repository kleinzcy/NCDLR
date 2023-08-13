import os
import random
import warnings
import torch.nn as nn
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from utils import get_net_builder, get_logger, get_port, over_write_args_from_file, ramps, AverageMeter, init_params
from data.get_dataset import get_discover_datasets
from utils.utils import split_cluster_acc_v2, cluster_acc, split_cluster_acc_v2_class
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from utils.sinkhorn_knopp import Balanced_sinkhorn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from utils.utils import model_statistics
from utils.sk_memory import SKMemory



def main(args):
    dataset = get_discover_datasets(args.dataset, args)
    train_dataset, unlabeled_train_dataset, test_dataset = dataset["train_dataset"], \
                                                           dataset['train_unlabel_dataset'], \
                                                           dataset["test_dataset"]
    class_unlabel_nums = dataset["class_unlabel_nums"]
    train_ulabeled_loader = torch.utils.data.DataLoader(
        unlabeled_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # estimate k
    if args.est_k:
        print("Note that we estimate K")
        args.num_unlabeled_classes += args.error
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    _net_builder = get_net_builder(args.net, args.net_from_name)

    model = _net_builder(num_classes=args.num_classes,
                         num_unseen_classes=args.num_unlabeled_classes,
                         num_seen_classes=args.num_labeled_classes,
                         pretrained=args.use_pretrain,
                         pretrained_path=args.pretrain_path)

    model = init_params(model, random_head=args.random_head)
    model_statistics(model)
    model = model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay_opt)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        warmup_start_lr=args.min_lr,
        eta_min=args.min_lr,
    )

    state = {k: v for k, v in args._get_kwargs()}
    wandb.init(project=args.project,
               entity=args.entity,
               config=state,
               name=args.save_name,
               dir=args.save_path)

    SK_memory = SKMemory(inputSize=args.num_unlabeled_classes, K=2048).cuda()
    SK_memory_bar = SKMemory(inputSize=args.num_unlabeled_classes, K=2048).cuda()

    balanced_sinkhorn = Balanced_sinkhorn(args, num_cluster=args.num_unlabeled_classes, SK_memory=SK_memory)
    # train
    start_epoch = 0
    best_scores = {
        "test/all/avg": 0,
        "test/seen/avg": 0,
        "test/novel/avg": 0,
        "epoch": 0
    }
    if args.resume:
        checkpoint = torch.load(args.load_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = checkpoint["scheduler"]

    for epoch in range(start_epoch, args.max_epochs):
        train(args, best_scores, model, SK_memory, SK_memory_bar,
              train_dataloader,
              train_ulabeled_loader, test_dataloader, epoch, optimizer,
              scheduler, wandb, class_unlabel_nums, balanced_sinkhorn)

        scheduler.step()


def mse(preds, targets):
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


def swapped_prediction(args, logits, targets):
    loss = 0
    for view in range(args.num_large_crops):
        for other_view in np.delete(range(args.num_crops), view):
            loss += mse(logits[other_view], targets[view])
    return loss / (args.num_large_crops * (args.num_crops - 1))


def train(args, best_scores, model, SK_memory, SK_memory_bar,
          train_dataloader,
          train_unlabeled_dataloadder, all_eval_loader, epoch, optimizer,
          scheduler, logger, class_unlabel_nums, balanced_sinkhorn):

    model.train()
    bar = tqdm(train_dataloader)

    for id, batch in enumerate(bar):

        optimizer.zero_grad()

        views_lab, labels_lab, _, views_unlab, labels_unlab, _ = batch
        images = [
            torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)
        ]
        labels = torch.cat([labels_lab, labels_unlab])

        images = [image.cuda() for image in images]
        labels = labels.cuda()
        label = labels.long()
        mask_lb = label < args.num_labeled_classes

        x = images[0]
        x_bar = images[1]
        x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
        outputs = model(x)
        outputs_bar = model(x_bar)

        logits = torch.cat([outputs['seen_logits'], outputs['unseen_logits']],
                           dim=-1)
        logits_bar = torch.cat(
            [outputs_bar["seen_logits"], outputs_bar["unseen_logits"]], dim=-1)


        prob = logits
        prob_bar = logits_bar

        targets_lab = F.one_hot(label[mask_lb],
                                num_classes=args.num_classes).float()
        SK_memory.initial(
            logits[~mask_lb, args.num_labeled_classes:].clone().detach(),
            labels[~mask_lb].float())
        SK_memory_bar.initial(
            logits_bar[~mask_lb, args.num_labeled_classes:].clone().detach(),
            labels[~mask_lb].float())

        # learn weight for different data to balance the distribution.
        targets_u = balanced_sinkhorn(
            logits=logits[~mask_lb, args.num_labeled_classes:].clone().detach(),
            args=args,
            num_cluster=args.num_unlabeled_classes,
            SK_memory=SK_memory,
            labels=labels[~mask_lb].float(),
            epoch=epoch).detach()
        targets_ubar = balanced_sinkhorn(
            logits=logits_bar[~mask_lb,
                   args.num_labeled_classes:].clone().detach(),
            args=args,
            num_cluster=args.num_unlabeled_classes,
            SK_memory=SK_memory_bar,
            labels=labels[~mask_lb].float(),
            epoch=epoch).detach()

        loss_ce = mse(prob[mask_lb], targets_lab) + mse(prob_bar[mask_lb], targets_lab)
        loss_u_ce = mse(prob[~mask_lb, args.num_labeled_classes:], targets_ubar) + mse(prob_bar[~mask_lb, args.num_labeled_classes:], targets_u)

        loss = (loss_ce + loss_u_ce) / 2

        loss.backward()
        optimizer.step()
        bar.set_postfix({"loss": "{:.2f}".format(loss.detach().cpu().numpy())})
        results = {
            "loss": loss.clone(),
            "lr": optimizer.param_groups[0]["lr"],
            "loss_u_ce": loss_u_ce,
            "loss_ce": loss_ce,
        }
        logger.log(results)

    lr = np.around(optimizer.param_groups[0]["lr"], 4)

    test_results = test(model, all_eval_loader, args, class_unlabel_nums, 'test')
    logger.log(test_results)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler},
               os.path.join(args.save_path, "latest_model.pth"))
    if best_scores['test/all/avg'] < test_results["test/all/avg"]:
        best_scores.update(test_results)
        best_scores['epoch'] = epoch
        torch.save(model.state_dict(),
                   os.path.join(args.save_path, "best_model.pth"))

    print("{}--Epoch-[{}/{}]--LR-[{}]--Best-Epoch-[{}]-All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]--Test-All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]"
        "--Head-[{:.2f}]--Medium-[{:.2f}]--Tail-[{:.2f}]--novel-nmi--[{:.2f}]--all-nmi--[{:.2f}]"
            .format(
            args.save_path,
            epoch,
            args.max_epochs,
            lr,
            best_scores["epoch"],
            best_scores['test/all/avg'] * 100,
            best_scores['test/novel/avg'] * 100,
            best_scores['test/seen/avg'] * 100,
            test_results["test/all/avg"] * 100,
            test_results["test/novel/avg"] * 100,
            test_results["test/seen/avg"] * 100,
            test_results["test/head/avg"] * 100,
            test_results["test/medium/avg"] * 100,
            test_results["test/tail/avg"] * 100,
            test_results["test/novel_nmi/avg"],
            test_results["test/all_nmi/avg"],
        ))


def test(model, test_loader, args, class_unlabel_nums, prefix):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        outputs = model(x)
        output = torch.cat([outputs['seen_logits'], outputs['unseen_logits']],
                           dim=-1)
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
    results = {}
    
    if args.dataset in ["herbarium_19", "inaturalist18"]:
        _res, _ = split_cluster_acc_v2_class(args, targets.astype(int), preds.astype(int),
                                       class_unlabel_nums=class_unlabel_nums,
                                       num_seen=args.num_labeled_classes)
    else:
        _res, _ = split_cluster_acc_v2(args, targets.astype(int), preds.astype(int), class_unlabel_nums=class_unlabel_nums,
                                num_seen=args.num_labeled_classes)

    for key, value in _res.items():
        if key in results.keys():
            results[key].append(value)
        else:
            results[key] = [value]
    log = {}
    for key, value in results.items():
        log[prefix + "/" + key + "/" + "avg"] = round(
            sum(value) / len(value), 4)
    return log


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='saved_models/UNO')
    parser.add_argument("--checkpoint_dir",
                        default="checkpoints",
                        type=str,
                        help="checkpoint dir")
    parser.add_argument("--dataset",
                        default="CIFAR100",
                        type=str,
                        help="dataset")
    parser.add_argument("--data_dir",
                        default="datasets",
                        type=str,
                        help="data directory")
    parser.add_argument("--log_dir",
                        default="logs",
                        type=str,
                        help="log directory")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="batch size")
    parser.add_argument("--num_workers",
                        default=4,
                        type=int,
                        help="number of workers")

    parser.add_argument("--lr", default=0.4, type=float, help="learning rate")
    parser.add_argument("--min_lr",
                        default=0.001,
                        type=float,
                        help="min learning rate")
    parser.add_argument('--amp',
                        default=False,
                        action="store_true",
                        help='use mixed precision training or not')
    parser.add_argument("--momentum_opt",
                        default=0.9,
                        type=float,
                        help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt",
                        default=1.5e-4,
                        type=float,
                        help="weight decay")
    parser.add_argument("--warmup_epochs",
                        default=5,
                        type=int,
                        help="warmup epochs")
    parser.add_argument("--max_epochs",
                        default=50,
                        type=int,
                        help="warmup epochs")
    parser.add_argument('--net', type=str, default='vit_small_patch2_32')
    parser.add_argument('--net_from_name', action="store_true", default=False)
    parser.add_argument('--use_pretrain', default=True, action='store_true')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='seed for initializing training. ')

    parser.add_argument("--arch",
                        default="resnet18",
                        type=str,
                        help="backbone architecture")
    parser.add_argument("--proj_dim",
                        default=256,
                        type=int,
                        help="projected dim")
    parser.add_argument("--hidden_dim",
                        default=2048,
                        type=int,
                        help="hidden dim in proj/pred head")
    parser.add_argument("--overcluster_factor",
                        default=1,
                        type=int,
                        help="overclustering factor")
    parser.add_argument("--num_heads",
                        default=5,
                        type=int,
                        help="number of heads for clustering")
    parser.add_argument("--num_hidden_layers",
                        default=1,
                        type=int,
                        help="number of hidden layers")
    parser.add_argument("--num_iters_sk",
                        default=3,
                        type=int,
                        help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk",
                        default=0.5,
                        type=float,
                        help="epsilon for the Sinkhorn")
    parser.add_argument("--temperature",
                        default=1,
                        type=float,
                        help="softmax temperature")
    parser.add_argument("--project",
                        default="MIv2",
                        type=str,
                        help="wandb project")
    parser.add_argument("--entity",
                        default="rikkixu",
                        type=str,
                        help="wandb entity")
    parser.add_argument("--offline",
                        default=True,
                        action="store_true",
                        help="disable wandb")
    parser.add_argument("--num_labeled_classes",
                        default=50,
                        type=int,
                        help="number of labeled classes")
    parser.add_argument("--num_unlabeled_classes",
                        default=50,
                        type=int,
                        help="number of unlab classes")
    parser.add_argument("--pretrain_path",
                        type=str,
                        help="pretrained checkpoint path")
    parser.add_argument("--multicrop",
                        default=False,
                        action="store_true",
                        help="activates multicrop")
    parser.add_argument("--num_large_crops",
                        default=2,
                        type=int,
                        help="number of large crops")
    parser.add_argument("--num_small_crops",
                        default=2,
                        type=int,
                        help="number of small crops")

    parser.add_argument("--ss_pretrained",
                        default=False,
                        action="store_true",
                        help="self-supervised pretrain")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu")
    parser.add_argument("--ratio",
                        type=float,
                        default=50,
                        help="the percentage of labeled data")
    parser.add_argument("--regenerate",
                        default=False,
                        action="store_true",
                        help="whether to generate data again")
    parser.add_argument("--resume",
                        default=False,
                        action="store_true",
                        help="whether to use old model")
    parser.add_argument("--save-model",
                        default=False,
                        action="store_true",
                        help="whether to save model")
    parser.add_argument("--imbalance-factor",
                        default=1,
                        type=int,
                        help="imbalance factor")
    parser.add_argument('-sn', '--save_name', type=str)

    parser.add_argument("--algorithm", default='UNO', type=str, help="")
    parser.add_argument("--lab_imbalance_factor", type=int, help="")
    parser.add_argument("--imbalance_factor", type=int, help="")
    parser.add_argument("--num_classes",
                        default=100,
                        type=int,
                        help="number of small crops")
    parser.add_argument("--random_head",
                        default=False,
                        action="store_true",
                        help="")

    parser.add_argument("--cosine_classifier", type=int, help="")

    parser.add_argument("--lr_w", type=float, help="")
    parser.add_argument("--confidence", type=float, help="")
    parser.add_argument("--set_imb", type=int, help="")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--num_outer_iters", type=int, help="")
    parser.add_argument("--flag", type=int, default=0, help="")
    parser.add_argument("--est_k",
                        type=bool,
                        default=False,
                        help="")

    # config file
    parser.add_argument('--c', type=str, default='')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    over_write_args_from_file(args, args.c)

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops
    return args


if __name__ == "__main__":
    args = get_config()
    device = torch.device("cuda" if args.cuda else "cpu")

    args.dump_path = os.path.join(
        "data/splits", f'{args.dataset}-labeled-{args.num_labeled_classes}'
                       f'-unlabeled-{args.num_unlabeled_classes}-'
                       f'imbalance-{args.imbalance_factor}.pkl')

    os.environ["WANDB_API_KEY"] = "Your wandb key"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)
    args.val = False
    save_path = os.path.join(args.save_dir, args.save_name)
    args.save_path = save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(args)
    main(args)