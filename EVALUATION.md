### Genral-purpose models
The general-purpose models are affine-invariant and as such need a pre-alignment step before an error can be computed.

Sample code for NYUv2 can be found here:
https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022

Sample code for KITTI can be found here:
https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d


### KITTI
* Remove images from `input/` and `output_monodepth/` folders
* Download `kitti_eval_dataset.zip` https://drive.google.com/file/d/1GbfMGuwg2VS06Vl75-_tB5FDj9EOrjl0/view?usp=sharing and unzip it in the `input/` folder (or follow this repository https://github.com/cogaplex-bts/bts to get RGB and Depth images from list [eigen_test_files_with_gt.txt](https://github.com/cogaplex-bts/bts/blob/master/train_test_inputs/eigen_test_files_with_gt.txt) )
* Download [dpt_hybrid_kitti-cb926ef4.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt) model and place it in the `weights/` folder
* `python run_monodepth.py --model_type dpt_hybrid_kitti --kitti_crop --absolute_depth --backbone hourglass_vitb_rn50_384 -l 4 -n 256`
* `python ./eval_with_pngs.py --pred_path ./output_monodepth/ --gt_path ./input/gt/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --garg_crop --do_kb_crop`

----

### NYUv2
* Remove images from `input/` and `output_monodepth/` folders
* Download `nyu_eval_dataset.zip` https://drive.google.com/file/d/1b37uu-bqTZcSwokGkHIOEXuuBdfo80HI/view?usp=sharing and unzip it in the `input/` folder (or follow this repository https://github.com/cogaplex-bts/bts to get RGB and Depth images from list [nyudepthv2_test_files_with_gt.txt](https://github.com/cogaplex-bts/bts/blob/master/train_test_inputs/nyudepthv2_test_files_with_gt.txt) )
* Download [dpt_hybrid_nyu-2ce69ec7.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt) model (**or a new model** that is fine-tuned with slightly different hyperparameters [dpt_hybrid_nyu_new-217f207d.pt](https://drive.google.com/file/d/1Nxv2OiqhAMosBL2a3pflamTW39dMjaSp/view?usp=sharing)  ) and place it in the `weights/` folder
* Download [eval_with_pngs.py](https://raw.githubusercontent.com/cogaplex-bts/bts/5a55542ebbe849eb85b5ce9592365225b93d8b28/utils/eval_with_pngs.py) in the root folder
* `python run_monodepth.py --model_type dpt_hybrid_nyu --absolute_depth --backbone hourglass_vitb_rn50_384 -l 6 -n 784`
(or **for new model** `python run_monodepth.py --model_type dpt_hybrid_nyu --absolute_depth --model_weights weights/dpt_hybrid_nyu_new-217f207d.pt --backbone hourglass_vitb_rn50_384 -l 6 -n 784` )
* `python ./eval_with_pngs.py --pred_path ./output_monodepth/ --gt_path ./input/gt/ --dataset nyu --max_depth_eval 10  --eigen_crop`
