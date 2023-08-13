import torch
from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
import os
import pickle as pkl
# from data.data_utils import subsample_instances
from config import cifar_10_root, cifar_100_root
import math
from data.utils import DiscoverTargetTransform


class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def get_cifar_datasets(transform_train, transform_val, num_labeled_classes,
                       num_unlabeled_classes, ratio, lab_imbalance_factor=1, dataset="CIFAR100", regenerate=False, imbalance_factor=1, val=False):
    if dataset == "CIFAR10":
        train_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_train)
        val_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_val)
        test_dataset = CustomCIFAR10(cifar_10_root, train=False, transform=transform_val)
        num_classes = 10
    elif dataset == "CIFAR100":
        train_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_train)
        val_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_val)
        test_dataset = CustomCIFAR100(cifar_100_root, train=False, transform=transform_val)
        num_classes = 100
    else:
        raise NotImplementedError("cifar10 or cifar100")
    np.random.seed(1)
    # split dataset
    assert num_unlabeled_classes == 50, "num_unlabeled_classes must be 50"
    dump_path = os.path.join("data/splits", f'{dataset}-labeled-{num_labeled_classes}'
                                            f'-unlabeled-{num_unlabeled_classes}-'
                                            f'imbalance-{imbalance_factor}.pkl')
    if regenerate or not os.path.exists(dump_path):
        shuffle_class_idx = np.arange(num_classes)
        np.random.shuffle(shuffle_class_idx)
        print(shuffle_class_idx)
        # print("the class idx don't be shuffled")
        target_transform = {n: o for o, n in enumerate(shuffle_class_idx)}
        labeled_classes = shuffle_class_idx[:num_labeled_classes]
        # unlabeled_class = shuffle_class_idx[num_labeled_classes:]

        print("Regenerate label file, num_labeled_classes: {}, num_unlabeled_classes: {}, imbalance_factor: "
                "{}-{}".format(num_labeled_classes, num_unlabeled_classes, lab_imbalance_factor, imbalance_factor))
        train_indices_lab = []
        train_indices_unlab = []
        nums = torch.zeros(100)
        for lc in range(num_labeled_classes):
            idx = np.nonzero(np.array(train_dataset.targets) == shuffle_class_idx[lc])[0]
            num = len(idx) * ((1 / lab_imbalance_factor) ** (lc / (num_labeled_classes - 1)))
            np.random.shuffle(idx)
            idx = idx[:int(num)]
            nums[lc] = num
            train_indices_lab.extend(idx)


        for lc in range(num_labeled_classes, num_classes):
            idx = np.nonzero(np.array(train_dataset.targets) == shuffle_class_idx[lc])[0]
            num = len(idx) * ((1 / imbalance_factor) ** ((lc - num_labeled_classes) / (num_classes - num_labeled_classes - 1)))
            np.random.shuffle(idx)
            idx = idx[:int(num)]
            nums[lc] = num
            train_indices_unlab.extend(idx)

        train_indices_unlab = np.array(train_indices_unlab)
        train_indices_lab = np.array(train_indices_lab)
        with open(dump_path, "wb") as f:
            pkl.dump({"lab": train_indices_lab, "unlab": train_indices_unlab, "target_transform": target_transform, "nums": nums, "labeled_classes": labeled_classes}, f)
    else:
        print(f"Loading from {dump_path}")
        with open(dump_path, "rb") as f:
            cache = pkl.load(f)
            train_indices_lab, train_indices_unlab, target_transform, labeled_classes, nums = cache["lab"], cache["unlab"], cache["target_transform"], cache["labeled_classes"], cache["nums"]
            print(target_transform.keys())

    train_dataset.target_transform = DiscoverTargetTransform(target_transform)
    val_dataset.target_transform = DiscoverTargetTransform(target_transform)
    test_dataset.target_transform = DiscoverTargetTransform(target_transform)

    train_label_dataset = torch.utils.data.Subset(train_dataset, train_indices_lab)
    val_label_dataset = torch.utils.data.Subset(val_dataset, train_indices_lab)
    train_unlabel_dataset = torch.utils.data.Subset(train_dataset, train_indices_unlab)
    val_unlabel_dataset = torch.utils.data.Subset(val_dataset, train_indices_unlab)


    test_indices_seen = np.where(np.isin(np.array(test_dataset.targets), labeled_classes))[0]
    test_seen_dataset = torch.utils.data.Subset(test_dataset, test_indices_seen)

    all_datasets = {"train_label_dataset": train_label_dataset, "train_unlabel_dataset": train_unlabel_dataset,
                    "val_label_dataset": val_label_dataset, "val_unlabel_dataset": val_unlabel_dataset,
                    "test_dataset": test_dataset,
                    "test_seen_dataset": test_seen_dataset,
                    "train_label_target": [target_transform[train_label_dataset.dataset.targets[i]] for i in train_indices_lab],
                    "class_unlabel_nums": nums}
    return all_datasets


if __name__ == "__main__":
    pass