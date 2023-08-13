import os

import torchvision
import numpy as np
from copy import deepcopy
import ipdb

from config import herbarium_dataroot


class HerbariumDataset19(torchvision.datasets.ImageFolder):

    def __init__(self, *args, **kwargs):

        # Process metadata json for training images into a DataFrame
        super().__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):

        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]

        return img, label, uq_idx


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]

    return dataset


def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [
        x for x, l in enumerate(dataset.targets) if l in include_classes
    ]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # Have a balanced test set
        v_ = np.random.choice(cls_idxs,
                              replace=False,
                              size=(val_instances_per_class, ))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_herbarium_datasets(train_transform,
                           test_transform,
                           num_labeled_classes=342,
                           num_unlabeled_classes=341,
                           seed=0,
                           split_train_val=False):

    np.random.seed(0)
    train_classes = np.random.choice(range(683, ),
                                     size=num_labeled_classes,
                                     replace=False)
    print(train_classes[:10])
    # Init entire training set
    train_dataset = HerbariumDataset19(transform=train_transform,
                                       root=os.path.join(
                                           herbarium_dataroot, 'small-train'))
    val_dataset = HerbariumDataset19(transform=test_transform,
                                       root=os.path.join(
                                           herbarium_dataroot, 'small-train'))

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    # TODO: Subsampling unlabelled set in uniform random fashion from training data, will contain many instances of dominant class
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset),
                                               include_classes=train_classes)
    # 我是随机选的几个作为labeled data，很有可能我选的全是tail，并且把target换成了1-342
    val_dataset_labelled = subsample_classes(deepcopy(val_dataset),
                                               include_classes=train_classes)

    unlabelled_indices = set(train_dataset.uq_idxs) - set(
        train_dataset_labelled.uq_idxs)

    train_dataset_unlabelled = subsample_dataset(
        deepcopy(train_dataset), np.array(list(unlabelled_indices)))
    val_dataset_unlabelled = subsample_dataset(
        deepcopy(val_dataset), np.array(list(unlabelled_indices)))

    # Get test dataset
    test_dataset = HerbariumDataset19(transform=test_transform,
                                      root=os.path.join(
                                          herbarium_dataroot,
                                          'small-validation'))

    # Transform dict
    unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[k] = i

    nums = {}
    for i in train_dataset_unlabelled.targets:
        if target_xform_dict[i] in nums:
            nums[target_xform_dict[i]] = nums[target_xform_dict[i]] + 1
        else:
            nums[target_xform_dict[i]] = 1
    tail = []
    medium = []
    head = []
    valu = sorted(nums.values())

    for (key, value) in nums.items():
        if value < valu[int(len(valu) * 0.3)]:
            tail.append(key)
        elif value >= valu[int(len(valu) * 0.3)] and value < valu[int(len(valu) * 0.7)]:
            medium.append(key)
        else:
            head.append(key)

    lab_nums = {}

    for i in train_dataset_labelled.targets:
        if target_xform_dict[i] in lab_nums:
            lab_nums[target_xform_dict[i]] = lab_nums[target_xform_dict[i]] + 1
        else:
            lab_nums[target_xform_dict[i]] = 1


    lab_tail = []
    lab_medium = []
    lab_head = []
    lab_valu = sorted(lab_nums.values())

    for (key, value) in lab_nums.items():
        if value < valu[int(len(lab_valu) * 0.3)]:
            lab_tail.append(key)
        elif value >= valu[int(len(lab_valu) * 0.3)] and value < valu[int(len(lab_valu) * 0.7)]:
            lab_medium.append(key)
        else:
            lab_head.append(key)


    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    val_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]

    # Either split train into train and val or use test set as val
    #train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    #val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_label_dataset': train_dataset_labelled,
        'train_unlabel_dataset': train_dataset_unlabelled,
        'val_label_dataset': val_dataset_labelled,
        'val_unlabel_dataset': val_dataset_unlabelled,
        'test_dataset': test_dataset,
        "class_unlabel_nums": {'head': head, 'med': medium, 'tail': tail},
        "class_labeled": {'head': lab_head, 'med': lab_medium, 'tail': lab_tail}
    }

    return all_datasets


if __name__ == '__main__':

    np.random.seed(0)
    # train_classes = np.random.choice(range(683, ),
    #                                  size=(int(683 / 2)),
    #                                  replace=False)

    x = get_herbarium_datasets(None, None)

    assert set(x['train_unlabelled'].targets) == set(range(683))

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(
        set.intersection(set(x['train_labelled'].uq_idxs),
                         set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(
        len(set(x['train_labelled'].uq_idxs)) +
        len(set(x['train_unlabelled'].uq_idxs)))
    print('Printing number of labelled classes...')
    print(len(set(x['train_labelled'].targets)))
    print('Printing total number of classes...')
    print(len(set(x['train_unlabelled'].targets)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')