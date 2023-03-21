import torch
import timm

from .tome.timm import apply_patch as apply_tome
from .vit import _make_vit_b_rn50_backbone


def _make_pretrained_tome_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
    tome_r=0.,
    **kwargs
):
    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    model = _make_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )

    apply_tome(model.model)
    model.model.r = tome_r
    return model

