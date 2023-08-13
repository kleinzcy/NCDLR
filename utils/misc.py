# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import ruamel.yaml as yaml
from torch.utils.tensorboard import SummaryWriter

def over_write_args_from_dict(args, dict):
    """
    overwrite arguments acocrding to config file
    """
    for k in dict:
        setattr(args, k, dict[k])


def over_write_args_from_file(args, yml):
    """
    overwrite arguments acocrding to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def send_model_cuda(args, model):
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


def count_parameters(model):
    # count trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name, use_tensorboard=False):
        self.tb_dir = tb_dir
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
            

    def update(self, tb_dict, it, suffix=None, mode="train"):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        if self.use_tensorboard:
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value, it)


