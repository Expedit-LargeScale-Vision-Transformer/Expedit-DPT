import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn
from timm.models.registry import register_model
from timm.models.vision_transformer_hybrid import _resnetv2, HybridEmbed, default_cfgs
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg

from functools import partial
import types
import math
import logging
from copy import deepcopy
import torch.nn.functional as F

from .vit import (
    Transpose,
    _resize_pos_embed,
    get_attention,
    attention,
    get_readout_oper,
)
from .cluster import TokenClusteringBlock
from .reconstruction import NaiveUnpooling, TokenReconstructionBlock

def reshape_as_aspect_ratio(x, ratio, channel_last=False):
    assert x.ndim == 3
    B, N, C = x.size()
    s = round(math.sqrt(N / (ratio[0] * ratio[1])))
    perm = (0, 1, 2) if channel_last else (0, 2, 1)
    
    return x.permute(*perm).view(B, C, s * ratio[0], s * ratio[1])

def get_aspect_ratio(x, y):
    gcd = math.gcd(x, y)
    return x // gcd, y // gcd

activations = {}
reconstructer = NaiveUnpooling().derive_unpooler()
_logger = logging.getLogger(__name__)

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

def get_activation_with_reconstruction(name):
    def hook(model, input, output):
        if output.shape[1] != reconstructer.num_features:
            out, _ = reconstructer.call(output[:, 1:])
            output = torch.cat([output[:, 0:1], out], dim=1)
        activations[name] = output

    return hook


class HourglassVisionTransformer(VisionTransformer):
    def __init__(
        self,
        cluster_after_output=True,
        clustering_location=0,
        num_cluster=256,
        temperture=0.1,
        cluster_iters=5,
        cluster_window_size=5,
        reconstruction_k=20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_clustering_layer = TokenClusteringBlock(
            num_spixels=num_cluster, 
            n_iters=cluster_iters,
            temperture=temperture,
            window_size=cluster_window_size,
        )
        self.cluster_after_output = cluster_after_output
        self.clustering_location = clustering_location

        self.token_reconstruction_layer = TokenReconstructionBlock(k=reconstruction_k, temperture=temperture)

    def cluster(self, x, aspect_ratio):
        reconstructer.update_state(feat_before_pooling=x[:, 1:])
        cls_tokens = x[:, 0:1]
        x = reshape_as_aspect_ratio(x[:, 1:], aspect_ratio)
        x, hard_labels = self.token_clustering_layer(x)
        x = torch.cat([cls_tokens, x], dim=1)
        reconstructer.update_state(hard_labels=hard_labels)
        reconstructer.update_state(feat_after_pooling=x)
        return x, hard_labels

    def forward_features(self, x):
        aspect_ratio = get_aspect_ratio(x.shape[-2], x.shape[-1])
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # x = self.blocks(x)
        global reconstructer
        reconstructer = self.token_reconstruction_layer.derive_unpooler()
        reconstructer.aspect_ratio = aspect_ratio
        reconstructer.num_features = x.shape[1]
        for i, block in enumerate(self.blocks):
            if not self.cluster_after_output and i == self.clustering_location:
                x, _ = self.cluster(x)
            x = block(x)
            if self.cluster_after_output and i == self.clustering_location:
                x, _ = self.cluster(x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

def forward_flex(self, x):
    b, c, h, w = x.shape
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

    # for blk in self.blocks:
    #     x = blk(x)
    reconstructer.aspect_ratio = aspect_ratio
    reconstructer.num_features = x.shape[1]
    for i, blk in enumerate(self.blocks):
        if not self.cluster_after_output and i == self.clustering_location:
            x, _ = self.cluster(x, aspect_ratio)
        x = blk(x)
        if self.cluster_after_output and i == self.clustering_location:
            x, _ = self.cluster(x, aspect_ratio)
            
    x = self.norm(x)

    return x

def _create_hourglass_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
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
        HourglassVisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

def _create_hourglass_vision_transformer_hybrid(variant, backbone, pretrained=False, **kwargs):
    default_cfg = deepcopy(default_cfgs[variant])
    embed_layer = partial(HybridEmbed, backbone=backbone)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return _create_hourglass_vision_transformer(
        variant, pretrained=pretrained, default_cfg=default_cfg, embed_layer=embed_layer, **kwargs)

@register_model
def hourglass_vit_base_r50_s16_384(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_hourglass_vision_transformer_hybrid(
        'vit_base_r50_s16_384', backbone=backbone, pretrained=pretrained, **model_kwargs)
    return model

@register_model
def hourglass_vit_base_resnet50_384(pretrained=False, **kwargs):
    # NOTE this is forwarding to model def above for backwards compatibility
    return hourglass_vit_base_r50_s16_384(pretrained=pretrained, **kwargs)

def _make_hourglass_vit_b_rn50_backbone(
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


def _make_pretrained_hourglass_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
    **kwargs
):
    model = timm.create_model("hourglass_vit_base_resnet50_384", pretrained=pretrained, **kwargs)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_hourglass_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )