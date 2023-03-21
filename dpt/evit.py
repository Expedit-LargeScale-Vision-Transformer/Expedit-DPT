""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman
# ------------------------------------------
# Modification:
# Added code for EViT training -- Copyright 2022 Youwei Liang
"""
import math
import logging
from functools import partial
from copy import deepcopy
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.helpers import build_model_with_cfg, build_model_with_cfg, overlay_external_default_cfg
from timm.models.registry import register_model
from timm.models.vision_transformer_hybrid import _resnetv2, HybridEmbed, default_cfgs
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn
from timm.models.layers import DropPath, Mlp

from .vit import (
    Transpose,
    _resize_pos_embed,
    get_attention,
    attention,
    get_readout_oper,
)
from .reconstruction import NaiveUnpooling, TokenReconstructionBlock

_logger = logging.getLogger(__name__)

def reshape_as_aspect_ratio(x, ratio, channel_last=False):
    assert x.ndim == 3
    B, N, C = x.size()
    # print('size',ratio,x.size())
    s = round(math.sqrt(N / (ratio[0] * ratio[1])))
    perm = (0, 1, 2) if channel_last else (0, 2, 1)
    
    return x.permute(*perm).view(B, C, s * ratio[0], s * ratio[1])

def get_aspect_ratio(x, y):
    gcd = math.gcd(x, y)
    return x // gcd, y // gcd

idxs = []
non_cls_list = []
activations = {}
reconstructer = NaiveUnpooling().derive_unpooler()
_logger = logging.getLogger(__name__)

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

def get_activation_with_reconstruction(name):
    def hook(model, input, output):
        output = output[0]
        output_cls = output[:, :1]
        out = output[:, 1:]
        if output.shape[1] != reconstructer.num_features:
            for i in reversed(range(len(idxs))):
                idx = idxs[i]
                non_cls = non_cls_list[i]
                _, N, C = non_cls.shape
                idx_compl = complement_idx(idx, N)
                reconstructer.update_state(feat_before_pooling = torch.gather(
                    non_cls, dim=1, index=idx_compl.unsqueeze(-1).expand(-1, -1, C)))
                reconstructer.update_state(feat_after_pooling = torch.gather(
                    non_cls, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, C)))
                out_non_cls, _ = reconstructer.call(out[:, :-1] if model.fuse_token else out)
                out_tmp = torch.empty_like(non_cls)
                out_tmp[0, idx] = out[:, :-1] if model.fuse_token else out
                out_tmp[0, idx_compl] = out_non_cls[0]
                out = out_tmp
            # out, _ = reconstructer.call(output[:, 1:])
            output = torch.cat([output_cls, out], dim=1)
        activations[name] = output

    return hook



def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


class EAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None, tokens=None):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        left_tokens = N - 1
        if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
            # index = torch.cat([cls_idx, idx + 1], dim=1)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens

        return  x, None, None, None, left_tokens


class EBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=0.,
                 fuse_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, keep_rate=keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_rate = keep_rate
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False):
        if keep_rate is None:
            keep_rate = self.keep_rate  # this is for inference, use the default keep rate
        B, N, C = x.shape

        tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
        x = x + self.drop_path(tmp)

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if get_idx and index is not None:
            return x, n_tokens, non_cls, idx
        return x, n_tokens, None, None


class EViT(VisionTransformer):
    """ EViT """
    
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
        base_keep_rate = 0.6,
        drop_loc = (3, 6, 9),
        fuse_token=True,
        reconstruction_teu=0.1,
        reconstruction_k=20,
    ):
        super().__init__(
            img_size, 
            patch_size, 
            in_chans, 
            num_classes, 
            embed_dim, 
            depth, 
            num_heads, 
            mlp_ratio, 
            qkv_bias, 
            representation_size, 
            distilled, 
            drop_rate, 
            attn_drop_rate, 
            drop_path_rate, 
            embed_layer, 
            norm_layer, 
            act_layer, 
            weight_init,
        )
        self.depth = depth
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        keep_rate = [1] * depth      
        for loc in drop_loc:
            keep_rate[loc] = base_keep_rate
        
        self.keep_rate = keep_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            EBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                keep_rate=keep_rate[i], fuse_token=fuse_token)
            for i in range(depth)])
        
        self.init_weights(weight_init)
        
        global reconstructer
        reconstructer = TokenReconstructionBlock(
            k=reconstruction_k, 
            temperture=reconstruction_teu
            ).derive_unpooler()

    @property
    def name(self):
        return "EViT"

def forward_flex(self, x, keep_rate=None, tokens=None, get_idx=True):
    b, c, h, w = x.shape
    if keep_rate is None:
        keep_rate = self.keep_rate
    if not isinstance(keep_rate, (tuple, list)):
        keep_rate = (keep_rate, ) * self.depth
    if not isinstance(tokens, (tuple, list)):
        tokens = (tokens, ) * self.depth
    assert len(keep_rate) == self.depth
    assert len(tokens) == self.depth
    aspect_ratio = get_aspect_ratio(h, w)

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    global idxs, non_cls_list
    left_tokens = []
    idxs = []
    non_cls_list = []
    reconstructer.aspect_ratio = aspect_ratio
    reconstructer.num_features = x.shape[1]
    for i, blk in enumerate(self.blocks):
        x, left_token, non_cls, idx = blk(x, keep_rate[i], tokens[i], get_idx)
        left_tokens.append(left_token)
        if idx is not None:
            idxs.append(idx)
        if non_cls is not None:
            non_cls_list.append(non_cls)
        
    x = self.norm(x)

    return x


def _create_e_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
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
        EViT, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

def _create_e_vision_transformer_hybrid(variant, backbone, pretrained=False, **kwargs):
    default_cfg = deepcopy(default_cfgs[variant])
    embed_layer = partial(HybridEmbed, backbone=backbone)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return _create_e_vision_transformer(
        variant, pretrained=pretrained, default_cfg=default_cfg, embed_layer=embed_layer, **kwargs)

@register_model
def e_vit_base_r50_s16_384(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_e_vision_transformer_hybrid(
        'vit_base_r50_s16_384', backbone=backbone, pretrained=pretrained, **model_kwargs)
    return model

@register_model
def e_vit_base_resnet50_384(pretrained=False, **kwargs):
    # NOTE this is forwarding to model def above for backwards compatibility
    return e_vit_base_r50_s16_384(pretrained=pretrained, **kwargs)

def _make_e_vit_b_rn50_backbone(
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
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation_with_reconstruction("1"))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation_with_reconstruction("2"))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
            get_activation("1")
        )
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
            get_activation("2")
        )

    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation_with_reconstruction("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation_with_reconstruction("4"))

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

def _make_pretrained_e_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
    **kwargs
):
    model = timm.create_model("e_vit_base_resnet50_384", pretrained=pretrained, **kwargs)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_e_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
