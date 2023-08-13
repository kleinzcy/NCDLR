import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import random
import os
import json
from tqdm import tqdm
from config import iNaturalist18
from copy import deepcopy
# modified from https://github.com/jiequancui/ResLT/blob/3f6b0ad95223f3afc9b4a4cc9d208149d1744538/Inat/datasets/inaturalist2018.py


class iNaturalist18Dataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, root=iNaturalist18):
        self.samples = []
        self.targets = []
        self.transform = transforms
        with open(txt) as f:
            for line in f:
                self.samples.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.transform = transform
        self.target_transform = target_transform
        # self.class_data = [[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y = self.labels[i]
        #     self.class_data[y].append(i)

        # self.cls_num_list = [len(self.class_data[i])
        #                      for i in range(self.num_classes)]
        # sorted_classes = np.argsort(self.cls_num_list)
        # self.class_map = [0 for i in range(self.num_classes)]
        # for i in range(self.num_classes):
        #     self.class_map[sorted_classes[i]] = i

        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.samples[index]
        label = self.targets[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return sample, label, self.uq_idxs[index]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    # dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    # dataset.targets = [int(x) for x in dataset.targets]

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


def get_inaturelist18_datasets(train_transform,
                               test_transform,
                               num_labeled_classes=500,
                               num_unlabeled_classes=500,
                               seed=0,
                               split_train_val=False):
    np.random.seed(0)
    num_classes = 8142
    train_txt = os.path.join(iNaturalist18, "iNaturalist18_train.txt")
    val_txt = os.path.join(iNaturalist18, "iNaturalist18_val.txt")

    all_classes = np.random.choice(range(num_classes),
                                     size=num_labeled_classes + num_unlabeled_classes,
                                     replace=False)
    train_classes = all_classes[:num_labeled_classes]
    val_classes = all_classes[num_labeled_classes:]
    print(train_classes[:10])
    # Init entire training set
    train_dataset = iNaturalist18Dataset(train_txt, transform=train_transform)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    # TODO: Subsampling unlabelled set in uniform random fashion from training data, will contain many instances of dominant class
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset),
                                               include_classes=train_classes)
    val_dataset_labelled = subsample_classes(deepcopy(train_dataset),
                                             include_classes=train_classes)

    train_dataset_unlabelled = subsample_classes(deepcopy(train_dataset), include_classes=val_classes)
    val_dataset_unlabelled = subsample_classes(deepcopy(train_dataset), include_classes=val_classes)

    val_dataset_labelled.transform = test_transform
    val_dataset_unlabelled.transform = test_transform

    # Get test dataset
    test_dataset = iNaturalist18Dataset(val_txt, transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=all_classes)
    # Transform dict
    target_xform_dict = {}
    for i, k in enumerate(all_classes):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    val_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    # head, medium, tail
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


if __name__ == "__main__":
    root = iNaturalist18
    json2txt = {
        'train2018.json': 'iNaturalist18_train.txt',
        'val2018.json': 'iNaturalist18_val.txt'
    }

    def convert(json_file, txt_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        lines = []
        for i in tqdm(range(len(data['images']))):
            assert data['images'][i]['id'] == data['annotations'][i]['id']
            img_name = data['images'][i]['file_name']
            label = data['annotations'][i]['category_id']
            lines.append(img_name + ' ' + str(label) + '\n')

        with open(txt_file, 'w') as ftxt:
            ftxt.writelines(lines)

    for k, v in json2txt.items():
        print('===> Converting {} to {}'.format(k, v))
        srcfile = os.path.join(root, k)
        convert(srcfile, v)
