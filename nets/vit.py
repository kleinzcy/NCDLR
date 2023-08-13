# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple
import numpy as np


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    try:
        orig_state_dict = checkpoint['model']
    except KeyError:
        orig_state_dict = checkpoint

    new_state_dict = {}
    for key, item in orig_state_dict.items():

        if key.startswith('module'):
            key = '.'.join(key.split('.')[1:])
        if key.startswith('fc') or key.startswith('classifier') or key.startswith('mlp') or key.startswith('head'):
            continue
        new_state_dict[key] = item

    match = model.load_state_dict(new_state_dict, strict=False)
    print(match)
    return model


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        assert (
            W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        self.expand = False
        if feat_in < num_classes:
            print("Warning: feature dimension is smaller than number of classes, ETF can not be initialized. We expand the dimension of feature.")
            self.expand = True
            expand_dim = feat_in
            while expand_dim < num_classes:
                expand_dim = expand_dim * 2
            self.fc = nn.Linear(feat_in, expand_dim)
            feat_in = expand_dim
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * \
            torch.matmul(P, I-((1/num_classes) * one))
        try:
            self.ori_M = M.cuda()
        except RuntimeError:
            self.ori_M = M
        self.LWS = LWS
        self.reg_ETF = reg_ETF
        if LWS:
            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
        else:
            self.learned_norm = torch.ones(1, num_classes).cuda()

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        # feat in has to be larger than num classes.
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(
            num_classes), atol=1e-06), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        if self.expand:
            x = self.fc(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        output = torch.matmul(x, self.ori_M)

        return output


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, num_unseen_classes=50,
            num_seen_classes=50, global_pool='token', 
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.grad_checkpointing = False
        self.embedding = nn.Linear(self.embed_dim, 128, bias=False)

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.num_features = self.embed_dim
        self.num_seen_classes = num_seen_classes
        self.num_unseen_classes = num_unseen_classes

        print("ETF Classifier.")
        self.head = ETF_Classifier(self.embed_dim, num_unseen_classes + num_seen_classes)

    def extract(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        x = self.extract(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        # em_out = self.embedding(x)

        output = self.head(x)
        output_seen, output_unseen = output[:, :self.num_seen_classes], output[:, self.num_seen_classes:]

        result_dict = {'seen_logits': output_seen, 'unseen_logits': output_unseen, 'feat': x}
        return result_dict

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def group_matcher(self, coarse=False, prefix=''):
        return dict(
            stem=r'^{}cls_token|{}pos_embed|{}patch_embed'.format(
                prefix, prefix, prefix),  # stem and embed
            blocks=[(r'^{}blocks\.(\d+)'.format(prefix), None),
                    (r'^{}norm'.format(prefix), (99999,))]
        )


def vit_tiny_patch2_32(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    model_kwargs = dict(img_size=32, patch_size=2, embed_dim=192,
                        depth=12, num_heads=3, drop_path_rate=0.1, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    # return model
    return model


def vit_small_patch2_32(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Small (ViT-S/2)
    """
    model_kwargs = dict(img_size=32, patch_size=2, embed_dim=384,
                        depth=12, num_heads=6, drop_path_rate=0.2, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def vit_small_patch16_224(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12,
                        num_heads=6, drop_path_rate=0.2, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def vit_base_patch16_96(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(img_size=96, patch_size=16, embed_dim=768,
                        depth=12, num_heads=12, drop_path_rate=0.2, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def vit_base_patch16_224(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12,
                        num_heads=12, drop_path_rate=0.2, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
