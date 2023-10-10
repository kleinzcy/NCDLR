from cmath import log
import torch
import torch.nn.functional as F
import pickle as pkl
from utils import get_logger
import torch.nn as nn
import ipdb
import math
import numpy as np
import os


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.5):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()  # n, 50 --2000 50
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class Imbalanced_SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.5, batch_size=256, num_cluster=50):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.idx = 0
        self.num_cluster = num_cluster
        self.order = None
        self.w0 = torch.ones(1, self.num_cluster) / self.num_cluster
        self.w0 = self.w0.cuda()
        # due to sigmoid, we set learn_k as 5.
        self.learn_k = torch.nn.Parameter(torch.ones(1) * 5)
        self.w = torch.tensor([n / (self.num_cluster - 1) for n in range(self.num_cluster)], requires_grad=True).cuda().reshape(1, -1)

    def forward(self, logits, labels, SK_memory, is_update):
        memory_logits, idx = SK_memory(logits, labels, is_update)
        if idx == 0:
            self.idx = SK_memory.queueSize
        else:
            self.idx = idx

        memory_logits = memory_logits - memory_logits.max()
        Q = torch.exp(memory_logits / self.epsilon).t()
        assert torch.sum(torch.isnan(Q)) == 0, "There is nan value in Q."

        K = torch.softmax(self.w * torch.log(torch.sigmoid(self.learn_k)), dim=1)
        B = Q.shape[1]
        # Sort Q
        self.order = torch.argsort(torch.sum(Q, dim=1), descending=True)
        Q = Q[self.order]
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q = (Q.t() * K).t()

            Q = torch.clamp(Q, min=1e-16)
            q_sum = torch.sum(Q, dim=0, keepdim=True)
            Q /= q_sum
            Q /= B
            assert torch.sum(torch.isnan(Q)) == 0, "There is nan value in Q."

        Q *= B
        # reorder
        reorder = torch.argsort(self.order, dim=0)
        # print(self.order)
        Q = Q[reorder]
        return Q.t()

    def get_w(self):
        K = torch.log_softmax(self.w * torch.log(torch.sigmoid(self.learn_k)), dim=1)
        return K

    def get_w0(self):
        K = torch.softmax(-self.w * np.log(np.exp(-5) + 1), dim=1)
        return K


class Balanced_sinkhorn(torch.nn.Module):
    def __init__(self, args, num_cluster=50, SK_memory=None):
        super().__init__()
        self.sk = Imbalanced_SinkhornKnopp(batch_size=SK_memory.queueSize, epsilon=args.epsilon_sk,
                                           num_cluster=num_cluster).cuda()
        self.opt = torch.optim.SGD(params=self.sk.parameters(), lr=args.lr_w, momentum=0.99)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits, args, SK_memory=None, labels=None):
        is_update = True

        for _ in range(args.num_outer_iters):
            Q = self.sk(logits, labels, SK_memory, is_update=is_update)

            preds = SK_memory.memory
            loss = - torch.mean(torch.sum(Q * preds, dim=1))

            input = self.sk.get_w().t()
            target = self.sk.get_w0().t()
            reg = self.kl_loss(input, target)

            total_loss = loss + args.gamma * reg

            self.opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sk.parameters(), 1)
            self.opt.step()
            is_update = False

        # sk.update_w()
        idx = self.sk.idx
        current_Q = Q[idx - logits.size(0):idx]

        return current_Q
