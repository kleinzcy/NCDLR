from data.utils import DiscoverDataset, TransformTwice
from data.cifar import get_cifar_datasets
from data.Imagenet100 import get_imagenet100_datasets
from data.herbarium import get_herbarium_datasets
from data.inature import get_inaturelist18_datasets
import functools
import pickle as pkl
import os
import torchvision.transforms as T


get_dataset_funcs = {
    'CIFAR10': functools.partial(get_cifar_datasets, dataset="CIFAR10"),
    'CIFAR100': functools.partial(get_cifar_datasets, dataset="CIFAR100"),
    'ImageNet100': get_imagenet100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'inaturalist18': get_inaturelist18_datasets

}


def get_transformation():
    transform_train = TransformTwice(T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    transform_val = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    return transform_train, transform_val


def get_discover_datasets(dataset_name, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError(
            f"dataset_name:{dataset_name}, not in {list(get_dataset_funcs.keys())}")

    transform_train, transform_val = get_transformation()

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    if dataset_name in ["herbarium_19", "inaturalist18"]:
        datasets = get_dataset_f(
            transform_train, transform_val, args.num_labeled_classes, args.num_unlabeled_classes)
    else:
        datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes,
                                 args.num_unlabeled_classes, args.ratio, regenerate=args.regenerate,
                                 imbalance_factor=args.imbalance_factor, lab_imbalance_factor=args.lab_imbalance_factor, val=args.val)

    # Train split (labelled and unlabelled classes) for training
    datasets["train_dataset"] = DiscoverDataset(
        datasets['train_label_dataset'], datasets['train_unlabel_dataset'])

    datasets["val_dataset"] = DiscoverDataset(
        datasets['val_label_dataset'], datasets['val_unlabel_dataset'])
    print("Lens of train dataset: {}, lens of unlab dataset: {}, lens of test dataset: {}".
          format(len(datasets["train_label_dataset"]), len(datasets['train_unlabel_dataset']), len(datasets["test_dataset"])))
    return datasets
