# _*_ coding: utf-8 _*_
"""
Time:     2022/10/18 13:10
Author:   Ruijie Xu
File:     sk_memory.py
"""
import torch
from torch import nn
import math
import torch.nn.functional as F


class SKMemory(nn.Module):
    """Memory Module for Sinkhorn Knopp"""

    def __init__(self, inputSize=50, K=2048, T=0.05, unlabel_num=50):
        super(SKMemory, self).__init__()
        self.inputSize = inputSize  # feature dim
        self.queueSize = K  # memory size
        self.T = T
        self.index = 0
        self.is_init = True
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv)),
        self.register_buffer('labels', torch.rand(self.queueSize))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, input_logits, labels, is_update=True):
        batchSize = input_logits.shape[0]
        if is_update:
            idx = self.update_memory(batchSize, input_logits, labels)
        else:
            idx = self.index
        return self.memory, idx

    def update_memory(self, batchSize, input_logits, labels):
        # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, input_logits)
            self.labels.index_copy_(0, out_ids, labels)
            self.index = (self.index + batchSize) % self.queueSize
            return self.index

    def initial(self, input_logits, labels):
        if self.is_init:
            assert input_logits.size(1) == self.memory.size(1), "the size of input_logits does not match the size of memory"
            r = int(self.queueSize / input_logits.size(0))
            self.memory[:] = input_logits.repeat(r, 1)
            self.labels[:] = labels.repeat(r)
            self.is_init = False