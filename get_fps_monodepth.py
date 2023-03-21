"""Compute depth maps for images in the input folder.
"""
import os
import torch
import cv2
import argparse
import time

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchprofile.torchprofile import profile_macs
from dpt.tome.timm import apply_patch as apply_tome

#from util.misc import visualize_attention
from util.parser import get_default_parser, get_model_args

import warnings
warnings.filterwarnings("ignore")

def find_max_batch_size(model,
                        min_batch: int,
                        max_batch: int,
                        device,
                        data_shape: list,
                        batch_dim: int,
                        optimize
                        ):
    # model = model.to(device)
    left = min_batch
    right = max_batch + 1

    with torch.no_grad():
        while left < right:
            torch.cuda.empty_cache()
            mid = (left + right) // 2
            data_shape[batch_dim] = mid
            # input_data = torch.rand(*data_shape).to(device)
            input_data = torch.rand(*data_shape,
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )

            if optimize == True and device == torch.device("cuda"):
                input_data = input_data.to(memory_format=torch.channels_last)
                input_data = input_data.half()

            try:
                model.forward(input_data)
            except RuntimeError:
                # out of memory
                # mid > best
                right = mid
            else:
                # mid <= best
                left = mid + 1
            del input_data
    return left - 1

def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True, input_scale=1.0, backbone=None, batchsize=None, **kwargs):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
    elif model_type == "dpt_hybrid_kitti":
        net_w = int(1216 * input_scale)
        net_h = int(352 * input_scale)

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384" if backbone is None else backbone,   # "cond_vitb_rn50_384"
            non_negative=True,
            enable_attention_hooks=False,
            **kwargs
        )
    elif model_type == "dpt_hybrid_nyu":
        net_w = int(640 * input_scale)
        net_h = int(480 * input_scale)

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384" if backbone is None else backbone,  # "cond_vitb_rn50_384"
            non_negative=True,
            enable_attention_hooks=False,
            **kwargs
        )
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    input_shape = (3, net_h, net_w)

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    # apply_tome(model.pretrained.model)
    # model.pretrained.model.r = tome_r
    model.eval()
    model.to(device)

    if batchsize == None:
        max_batch_size = find_max_batch_size(
            model, 
            min_batch=1, 
            max_batch=100,
            device=device,
            data_shape=[1, *input_shape],
            batch_dim=0,
            optimize=optimize
        )
        print('max batch size: ', max_batch_size)
        max_batch_size = int(max_batch_size * 0.8 + 0.5)
    else:
        max_batch_size = batchsize

    macs = 0
    num_warmup = 50
    num_samples = 200
    log_interval = 50
    pure_inf_time = 0
    print("start processing")
    for i in range(num_samples):
        sample = torch.rand([max_batch_size, *input_shape],
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
            
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        # compute
        start_time = time.perf_counter()   
        with torch.no_grad():
            prediction = model.forward(sample)
            # model.pretrained.model.forward_flex(sample)
        elapsed = time.perf_counter() - start_time

        # macs += profile_macs(model, sample)

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) * max_batch_size / pure_inf_time
                print('Done iterations [{:3}/ {}], '.format(i+1, num_samples) + 
                        'fps: {:.2f} img / s'.format(fps))

    print("finished")
    # vit = model.pretrained.model
    # vit.forward = vit.forward_flex
    print("vit macs: {:.4g} G".format(profile_macs(model, sample[:1]) / 1e9))
    # print("macs: {:.4g} G".format(macs / num_samples / 1e9))
    fps = (num_samples - num_warmup) * max_batch_size / pure_inf_time
    print('Overall fps: {:.2f} img / s'.format(fps))


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    kwargs = get_model_args(args)
    kwargs['resize_short_edge_length'] = args.resize_short_edge_length
    kwargs['backbone'] = args.backbone


    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
        batchsize=1,
        **kwargs
    )
    
    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
        **kwargs
    )
