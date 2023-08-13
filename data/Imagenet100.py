# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : Imagenet100.py
# DATE : 2022/8/28 13:00

# 1. Generate data split
# 2. Supervised Pretrain dataloader
# 3. Discovery dataloader
import torch
import numpy as np
import os
import pickle as pkl
# from data.data_utils import subsample_instances
from config import imagenet_root
import torch.utils.data as data
from PIL import Image
import glob


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageNetDataset(data.Dataset):

    def __init__(self, root, anno_file, loader=default_loader, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        filenames = []
        targets = []
        with open(anno_file, 'r') as fin:
            for line in fin.readlines():
                line_split = line.strip('\n').split(' ')
                filenames.append(line_split[0])
                targets.append(int(line_split[1]))

        self.samples = filenames
        self.targets = targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = os.path.join(self.root, self.samples[index])
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

    def __len__(self):
        return len(self.targets)


def get_imagenet100_datasets(transform_train, transform_val, num_labeled_classes,
                             num_unlabeled_classes, ratio, regenerate=False, lab_imbalance_factor=1,imbalance_factor=1, val=False):
    train_data_dir = os.path.join(imagenet_root, "ImgNetTrain")
    val_data_dir = os.path.join(imagenet_root, "ImgNetVal")

    if regenerate or not os.path.exists("./data/splits/ImageNet100_nums_unlabel_%d_imbalance_%d.pkl"):
        nums = generate(num_labeled_classes,num_unlabeled_classes,lab_imbalance_factor, imbalance_factor)
    else:
        with open("./data/splits/ImageNet100_nums_unlabel_%d_imbalance_%d.pkl", "rb") as f:
            nums = pkl.load(f)

    val_anno_file = './data/splits/ImageNet100_val.txt'.format(num_labeled_classes, imbalance_factor)
    if val:
        train_anno_file = './data/splits/ImageNet100_label_{:.0f}_imbalance_{:.0f}_val.txt'.format(num_labeled_classes, lab_imbalance_factor)
        train_unlab_anno_file = './data/splits/ImageNet100_unlabel_{:.0f}_imbalance_{:.0f}_val.txt'.format(num_labeled_classes,
                                                                                                imbalance_factor)
    else:
        train_anno_file = './data/splits/ImageNet100_label_{:.0f}_imbalance_{:.0f}.txt'.format(num_labeled_classes, lab_imbalance_factor)
        train_unlab_anno_file = './data/splits/ImageNet100_unlabel_{:.0f}_imbalance_{:.0f}.txt'.format(num_labeled_classes,
                                                                                                imbalance_factor)
    train_label_dataset = ImageNetDataset(train_data_dir, train_anno_file, transform=transform_train)
    train_unlabel_dataset = ImageNetDataset(train_data_dir, train_unlab_anno_file, transform=transform_train)
    val_unlab_dataset = ImageNetDataset(train_data_dir, train_unlab_anno_file, transform=transform_val)
    val_label_dataset = ImageNetDataset(train_data_dir, train_anno_file, transform=transform_val)
    test_dataset = ImageNetDataset(val_data_dir, val_anno_file, transform=transform_val)

    labeled_classes = np.arange(num_labeled_classes)

    # seen class
    val_indices_seen = np.where(np.isin(np.array(val_unlab_dataset.targets), labeled_classes))[0]
    val_seen_dataset = torch.utils.data.Subset(val_unlab_dataset, val_indices_seen)

    # testset
    test_indices_seen = np.where(np.isin(np.array(test_dataset.targets), labeled_classes))[0]
    test_seen_dataset = torch.utils.data.Subset(test_dataset, test_indices_seen)

    all_datasets = {"train_label_dataset": train_label_dataset, "train_unlabel_dataset": train_unlabel_dataset,
                    "val_unlabel_dataset": val_unlab_dataset, "val_label_dataset": val_label_dataset, "val_seen_dataset": val_seen_dataset,
                    "test_dataset": test_dataset, "test_seen_dataset": test_seen_dataset,
                    "class_unlabel_nums": nums
                    }
    return all_datasets


def generate(num_labeled_classes,num_unlabeled_classes,lab_imbalance_factor, imbalance_factor):
    folders = 'n01558993 n01601694 n01669191 n01751748 n01755581 n01756291 n01770393 n01855672 n01871265 n02018207 ' \
              'n02037110 n02058221 n02087046 n02088632 n02093256 n02093754 n02094114 n02096177 n02097130 n02097298 ' \
              'n02099267 n02100877 n02104365 n02105855 n02106030 n02106166 n02107142 n02110341 n02114855 n02120079 ' \
              'n02120505 n02125311 n02128385 n02133161 n02277742 n02325366 n02364673 n02484975 n02489166 n02708093 ' \
              'n02747177 n02835271 n02906734 n02909870 n03085013 n03124170 n03127747 n03160309 n03255030 n03272010 ' \
              'n03291819 n03337140 n03450230 n03483316 n03498962 n03530642 n03623198 n03649909 n03710721 n03717622 ' \
              'n03733281 n03759954 n03775071 n03814639 n03837869 n03838899 n03854065 n03929855 n03930313 n03954731 ' \
              'n03956157 n03983396 n04004767 n04026417 n04065272 n04200800 n04209239 n04235860 n04311004 n04325704 ' \
              'n04336792 n04346328 n04380533 n04428191 n04443257 n04458633 n04483307 n04509417 n04515003 n04525305 ' \
              'n04554684 n04591157 n04592741 n04606251 n07583066 n07613480 n07693725 n07711569 n07753592 n11879895'
    train_path = "/public/home/zhangchy2/workdir/data/NCD/ILSVRC2012/ImgNetTrain"
    val_path = "/public/home/zhangchy2/workdir/data/NCD/ILSVRC2012/ImgNetVal"
    IMG_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
    np.random.seed(0)
    cls_num = 50

    fout_train_label = open('data/splits/ImageNet100_label_%d_imbalance_%d.txt' % (num_labeled_classes, lab_imbalance_factor), 'w')
    fout_train_unlabel = open('data/splits/ImageNet100_unlabel_%d_imbalance_%d.txt' % (num_unlabeled_classes, imbalance_factor), 'w')
    fout_train_label_val = open('data/splits/ImageNet100_label_%d_imbalance_%d_val.txt' % (num_labeled_classes, lab_imbalance_factor), 'w')
    fout_train_unlabel_val = open('data/splits/ImageNet100_unlabel_%d_imbalance_%d_val.txt' % (num_unlabeled_classes, imbalance_factor), 'w')
    fout_val = open('data/splits/ImageNet100_val.txt', 'w')

    folders = folders.split(' ')
    nums = torch.zeros(100)
    val_cls = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    for i, folder_name in enumerate(folders):
        files = []
        val_files = []
        for extension in IMG_EXTENSIONS:
            files.extend(glob.glob(os.path.join(train_path, folder_name, '*' + extension)))
            val_files.extend(glob.glob(os.path.join(val_path, folder_name, '*' + extension)))


        for filename in files:
            filepath = os.path.join(folder_name, filename.split("/")[-1])
            if i < cls_num:

                if np.random.rand() <= ((1 / lab_imbalance_factor) ** (i / (num_labeled_classes - 1))):
                    nums[i] = nums[i] + 1
                    fout_train_label.write('%s %d\n' % (filepath, i))
                    if i in val_cls:
                        fout_train_unlabel_val.write('%s %d\n' % (filepath, i))
                    else:
                        fout_train_label_val.write('%s %d\n' % (filepath, i))
            else:
                if np.random.rand() <= ((1 / imbalance_factor) ** ((i - num_labeled_classes) / (num_unlabeled_classes - 1))):

                    nums[i] = nums[i] + 1
                    fout_train_unlabel.write('%s %d\n' % (filepath, i))
                    fout_train_unlabel_val.write('%s %d\n' % (filepath, i))

        for filename in val_files:
            filepath = os.path.join(folder_name, filename.split("/")[-1])
            fout_val.write('%s %d\n' % (filepath, i))

    fout_train_label.close()
    fout_train_unlabel.close()
    fout_val.close()
    fout_train_label_val.close()
    fout_train_unlabel_val.close()
    print("--Data has been generated!!!--")
    with open("./data/splits/ImageNet100_nums_unlabel_%d_imbalance_%d.pkl", "wb") as f:
        pkl.dump(nums, f)
    return nums

if __name__ == '__main__':
    generate(50,50,50)