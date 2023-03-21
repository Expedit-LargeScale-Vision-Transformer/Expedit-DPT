# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from functools import partial
import types
import math
import time
import logging
from copy import deepcopy

import timm
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn
from timm.models.registry import register_model
from timm.models.vision_transformer_hybrid import _resnetv2, HybridEmbed, default_cfgs
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from ACT.ada_clustering_attention import AdaClusteringAttention
from .vit import (
    Transpose,
    _resize_pos_embed,
    get_attention,
    attention,
    get_readout_oper,
    get_activation,
    activations,
    forward_flex
)

class ACTAttention(Attention):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        attn_drop=0, 
        proj_drop=0,
        group_Q=True,
        group_K=False,
        q_hashes=32,
        k_hashes=32,
    ):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)

        self.attention = AdaClusteringAttention(group_Q, group_K, q_hashes, k_hashes, attention_dropout=attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (
            self.attention(
                q.reshape(-1, N, C // self.num_heads), 
                k.reshape(-1, N, C // self.num_heads), 
                v.reshape(-1, N, C // self.num_heads)
            )
            .reshape(B, self.num_heads, N, -1)
            .transpose(1, 2)
            .reshape(B, N, C)
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ACTBlock(Block):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4, 
        qkv_bias=False, 
        drop=0, 
        attn_drop=0, 
        drop_path=0, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm,
        group_Q=True,
        group_K=False,
        q_hashes=32,
        k_hashes=32,
    ):
        super().__init__(
            dim, 
            num_heads, 
            mlp_ratio, 
            qkv_bias, 
            drop, 
            attn_drop, 
            drop_path, 
            act_layer, 
            norm_layer,
        )

        self.attn = ACTAttention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop,
            group_Q=group_Q,
            group_K=group_K,
            q_hashes=q_hashes,
            k_hashes=k_hashes,
        )


class ACTVisionTransformer(VisionTransformer):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True, 
        representation_size=None, 
        distilled=False, 
        drop_rate=0, 
        attn_drop_rate=0, 
        drop_path_rate=0, 
        embed_layer=..., 
        norm_layer=None, 
        act_layer=None, 
        weight_init='', 
        group_Q=True,
        group_K=False,
        q_hashes=32,
        k_hashes=32,
        **kwargs
    ):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, representation_size, distilled, drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer, act_layer, weight_init)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        for i in range(len(self.blocks)):
            self.blocks[i] = ACTBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate,
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer, 
                act_layer=act_layer,
                group_Q=group_Q,
                group_K=group_K,
                q_hashes=q_hashes,
                k_hashes=k_hashes,
            )


_logger = logging.getLogger(__name__)

def _create_act_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        ACTVisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

def _create_act_vision_transformer_hybrid(variant, backbone, pretrained=False, **kwargs):
    default_cfg = deepcopy(default_cfgs[variant])
    embed_layer = partial(HybridEmbed, backbone=backbone)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return _create_act_vision_transformer(
        variant, pretrained=pretrained, default_cfg=default_cfg, embed_layer=embed_layer, **kwargs)

@register_model
def act_vit_base_r50_s16_384(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_act_vision_transformer_hybrid(
        'vit_base_r50_s16_384', backbone=backbone, pretrained=pretrained, **model_kwargs)
    return model

@register_model
def act_vit_base_resnet50_384(pretrained=False, **kwargs):
    # NOTE this is forwarding to model def above for backwards compatibility
    return act_vit_base_r50_s16_384(pretrained=pretrained, **kwargs)

def _make_act_vit_b_rn50_backbone(
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    use_vit_only=False,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()

    pretrained.model = model

    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
            get_activation("1")
        )
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
            get_activation("2")
        )

    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    if enable_attention_hooks:
        pretrained.model.blocks[2].attn.register_forward_hook(get_attention("attn_1"))
        pretrained.model.blocks[5].attn.register_forward_hook(get_attention("attn_2"))
        pretrained.model.blocks[8].attn.register_forward_hook(get_attention("attn_3"))
        pretrained.model.blocks[11].attn.register_forward_hook(get_attention("attn_4"))
        pretrained.attention = attention

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    if use_vit_only == True:
        pretrained.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        pretrained.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
    else:
        pretrained.act_postprocess1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        pretrained.act_postprocess2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_act_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
    **kwargs
):
    model = timm.create_model("act_vit_base_resnet50_384", pretrained=pretrained, **kwargs)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_act_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
