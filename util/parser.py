import argparse


def get_default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument(
        "--backbone",
        default="vitb_rn50_384",
        help="backbone with different methods",
    )

    parser.add_argument(
        "-r",
        "--resize_short_edge_length",
        type=int,
        default=-1,
        help="resize the inputs while keeping ratio of the height and width"
    )

    parser.add_argument(
        "--input_scale",
        type=float,
        default=1.0,
        help="num of clustes"
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    add_model_args(parser)

    return parser

def add_model_args(parser):
    add_hourglass_vit_args(parser)
    add_tome_args(parser)
    add_evit_args(parser)
    add_act_vit_args(parser)

def add_hourglass_vit_args(parser):
    parser.add_argument(
        "-l",
        "--clustering_location",
        type=int,
        default=-1,
        help="location of clustering, ranging from [0, num of layers of transformer)"
    )
    parser.add_argument(
        "-n",
        "--num_cluster",
        type=int,
        default=1000,
        help="num of clusters, no more than total number of features"
    )
    parser.add_argument(
        "--cluster_iters",
        type=int,
        default=5,
        help="num of iterations in clustering"
    )
    parser.add_argument(
        "--temperture",
        type=float,
        default=1.,
        help="temperture in clustering and reconstruction"
    )
    parser.add_argument(
        "--cluster_window_size",
        type=int,
        default=5,
        help="window size in clustering"
    )
    parser.add_argument(
        "--reconstruction_k",
        type=int,
        default=20,
        help="k in token reconstruction layer of hourglass vit"
    )


def add_tome_args(parser):
    parser.add_argument(
        "--tome_r",
        type=float,
        default=0.5,
        help="num of tokens of each step of merging, ranging from [0, num of tokens / 2)"
    )


def add_evit_args(parser):
    parser.add_argument("--base_keep_rate", type=float, default=0.7)
    parser.add_argument("--drop_loc", type=int, nargs='+', default=[3, 6, 9])
    parser.add_argument("--fuse_token", action='store_true', default=False)


def add_act_vit_args(parser):
    parser.add_argument(
        "--act_plug_in_index",
        type=int,
        default=0,
        help='act attention plug in index'
    )
    parser.add_argument(
        "--act_plug_out_index",
        type=int,
        default=-1,
        help='act attention plug out index'
    )
    parser.add_argument(
        "--act_group_q",
        type=bool,
        default=True,
        help='act query use grouping'
    )
    parser.add_argument(
        "--act_group_k",
        type=bool,
        default=False,
        help='act query use grouping'
    )
    parser.add_argument(
        "--act_q_hashes",
        type=int,
        default=32,
        help='act query hash times'
    )
    parser.add_argument(
        "--act_k_hashes",
        type=int,
        default=32,
        help='act key hash times'
    )


def get_model_args(args):
    if 'hourglass_vit' in args.backbone:
        return {
            "clustering_location":  args.clustering_location,
            "num_cluster":          args.num_cluster,
            "cluster_iters":        args.cluster_iters,
            "temperture":           args.temperture,
            "cluster_window_size":  args.cluster_window_size,
            "reconstruction_k":  args.reconstruction_k,
        }
    elif 'tome' in args.backbone:
        return {
            "tome_r":               args.tome_r,
        }
    elif 'e_vit' in args.backbone:
        return {
            "base_keep_rate":       args.base_keep_rate,
            "drop_loc":             args.drop_loc,
            "fuse_token":           args.fuse_token,
        }
    elif 'act' in args.backbone:
        return {
            "act_plug_in_index": args.act_plug_in_index,
            "act_plug_out_index": args.act_plug_out_index,
            "group_Q": args.act_group_q,
            "group_K": args.act_group_k,
            "q_hashes": args.act_q_hashes,
            "k_hashes": args.act_k_hashes,
        }
    elif 'smyrf' in args.backbone:
        return {
            "smyrf_plug_in_index": args.smyrf_plug_in_index,
            "smyrf_plug_out_index": args.smyrf_plug_out_index,
            "smyrf_n_hashes": args.smyrf_n_hashes,
            "smyrf_q_cluster_size": args.smyrf_q_cluster_size,
            "smyrf_k_cluster_size": args.smyrf_k_cluster_size,
        }
    elif 'ats' in args.backbone:
        return {
            "ats_blocks":           args.ats_blocks_indexes,
            "num_tokens":           args.ats_num_tokens,
            "drop_tokens":          args.ats_drop_tokens   
        }
    return dict()

