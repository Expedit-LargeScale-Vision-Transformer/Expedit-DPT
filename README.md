# Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning

## Introduction

This is the official implementation of the paper "[Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning](https://arxiv.org/abs/2210.01035)" on [DPT](https://arxiv.org/abs/2103.13413). 

We will also implement several token-reduction methods on DPT.

+ [EViT](https://github.com/youweiliang/evit)
+ [ACT](https://github.com/gaopengcuhk/SMCA-DETR/tree/main/Adaptive_Cluster_Transformer)
+ [ToMe](https://github.com/facebookresearch/ToMe)

## Results 

### KITTI

#### Ours

| Method     | Backbone   | $\alpha$ | h $\times$ w   | GFLOPs | FPS   | Throughput (im/s) | RMSE  |
| ---------- | ---------- | -------- | -------------- | ------ | ----- | ----------------- | ----- |
| DPT        | ViT-Hybrid | -        | 76 $\times$ 22 | 810.0  | 12.55 | 12.02             | 2.573 |
| DPT + Ours | ViT-Hybrid | 6        | 28 $\times$ 28 | 723.4  | 13.30 | 13.87             | 2.593 |
| DPT + Ours | ViT-Hybrid | 4        | 20 $\times$ 20 | 631.8  | 16.41 | 16.88             | 2.625 |
| DPT + Ours | ViT-Hybrid | 4        | 16 $\times$ 16 | 613.6  | 17.14 | 17.44             | 2.660 |

### EViT

| Method     | Backbone   | keep rate | GFLOPs | FPS   | Throughput (im/s) | RMSE  |
| ---------- | ---------- | --------- | ------ | ----- | ----------------- | ----- |
| DPT        | ViT-Hybrid | -         | 810.0  | 12.55 | 12.02             | 2.573 |
| DPT + EViT | ViT-Hybrid | 0.8       | 702.4  | 14.50 | 14.05             | 2.731 |
| DPT + EViT | ViT-Hybrid | 0.7       | 663.0  | 15.81 | 15.17             | 2.992 |
| DPT + EViT | ViT-Hybrid | 0.5       | 605.2  | 16.94 | 17.41             | 4.319 |

### ToMe

| Method     | Backbone   | r%   | GFLOPs | FPS   | Throughput (im/s) | RMSE  |
| ---------- | ---------- | ---- | ------ | ----- | ----------------- | ----- |
| DPT        | ViT-Hybrid | -    | 810.0  | 12.55 | 12.02             | 2.573 |
| DPT + ToMe | ViT-Hybrid | 20%  | 741.6  | 13.04 | 12.75             | 2.608 |
| DPT + ToMe | ViT-Hybrid | 30%  | 697.8  | 14.24 | 14.34             | 2.615 |
| DPT + ToMe | ViT-Hybrid | 50%  | 615.8  | 16.29 | 16.29             | 2.600 |

### ACT

| Method    | Backbone | q_hashes | GFLOPs | FPS   | Throughput (im/s) | RMSE  |
| --------- | -------- | -------- | ------ | ----- | ----------------- | ----- |
| DPT       | ViT-L/16 | -        | 810.0  | 12.55 | 12.02             | 2.573 |
| DPT + ACT | ViT-L/16 | 32       | 810.4  | 10.46 | 10.18             | 7.151 |
| DPT + ACT | ViT-L/16 | 24       | 808.8  | 10.50 | 10.63             | 7.141 |
| DPT + ACT | ViT-L/16 | 20       | 806.6  | 10.84 | 11.12             | 7.150 |
| DPT + ACT | ViT-L/16 | 16       | 801.0  | 11.00 | 11.81             | 7.118 |

### NYUv2

#### Ours

| Method     | Backbone   | $\alpha$ | h $\times$ w   | GFLOPs | FPS   | Throughput (im/s) | RMSE  |
| ---------- | ---------- | -------- | -------------- | ------ | ----- | ----------------- | ----- |
| DPT        | ViT-Hybrid | -        | 40 $\times$ 30 | 560.2  | 18.66 | 18.56             | 0.357 |
| DPT + Ours | ViT-Hybrid | 6        | 24 $\times$ 24 | 504.2  | 19.19 | 20.60             | 0.360 |
| DPT + Ours | ViT-Hybrid | 4        | 16 $\times$ 16 | 439.2  | 22.08 | 24.69             | 0.362 |
| DPT + Ours | ViT-Hybrid | 2        | 14 $\times$ 14 | 395.0  | 26.02 | 28.93             | 0.364 |

### EViT

| Method     | Backbone   | keep rate | GFLOPs | FPS   | Throughput (im/s) | RMSE  |
| ---------- | ---------- | --------- | ------ | ----- | ----------------- | ----- |
| DPT        | ViT-Hybrid | -         | 560.2  | 18.66 | 18.56             | 0.357 |
| DPT + EViT | ViT-Hybrid | 0.8       | 490.6  | 20.39 | 21.49             | 0.372 |
| DPT + EViT | ViT-Hybrid | 0.7       | 464.6  | 21.72 | 23.11             | 0.399 |
| DPT + EViT | ViT-Hybrid | 0.5       | 425.8  | 23.87 | 25.20             | 0.509 |

### ToMe

| Method     | Backbone   | r%   | GFLOPs | FPS   | Throughput (im/s) | RMSE  |
| ---------- | ---------- | ---- | ------ | ----- | ----------------- | ----- |
| DPT        | ViT-Hybrid | -    | 560.2  | 18.66 | 18.56             | 0.357 |
| DPT + ToMe | ViT-Hybrid | 20%  | 513.6  | 18.27 | 20.45             | 0.354 |
| DPT + ToMe | ViT-Hybrid | 30%  | 483.2  | 19.33 | 22.16             | 0.354 |
| DPT + ToMe | ViT-Hybrid | 50%  | 431.6  | 22.40 | 25.36             | 0.365 |

### ACT

| Method    | Backbone | q_hashes | GFLOPs | FPS   | Throughput (im/s) | RMSE  |
| --------- | -------- | -------- | ------ | ----- | ----------------- | ----- |
| DPT       | ViT-L/16 | -        | 560.2  | 18.66 | 18.56             | 0.357 |
| DPT + ACT | ViT-L/16 | 32       | 560.4  | 13.85 | 15.26             | 0.566 |
| DPT + ACT | ViT-L/16 | 24       | 559.4  | 14.03 | 15.95             | 0.567 |
| DPT + ACT | ViT-L/16 | 20       | 558.2  | 14.33 | 16.29             | 0.567 |
| DPT + ACT | ViT-L/16 | 16       | 555.4  | 14.81 | 17.13             | 0.567 |

### Setup 

1) Download the model weights and place them in the `weights` folder:


Monodepth:

- [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), [Mirror](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing)
- [dpt_large-midas-2f21e586.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt), [Mirror](https://drive.google.com/file/d/1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G/view?usp=sharing)

Segmentation:

 - [dpt_hybrid-ade20k-53898607.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-ade20k-53898607.pt), [Mirror](https://drive.google.com/file/d/1zKIAMbltJ3kpGLMh6wjsq65_k5XQ7_9m/view?usp=sharing)
 - [dpt_large-ade20k-b12dca68.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-ade20k-b12dca68.pt), [Mirror](https://drive.google.com/file/d/1foDpUM7CdS8Zl6GPdkrJaAOjskb7hHe-/view?usp=sharing)

2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.8, PyTorch 1.12.1, OpenCV 4.5.2.54, and timm 0.4.12

   To use ToMe, you can follow [INSTALL.md](https://github.com/facebookresearch/ToMe/blob/main/INSTALL.md) to install the ToMe library.

   To use Act-ViT, you can install the ACT library as follows:

   ```shell
   cd dpt/act
   python setup.py install
   ```
    

### Usage 

1) Place one or more input images in the folder `input`.

2) Run a monocular depth estimation model:

    ```shell
    python run_monodepth.py
    ```
    
    Or run a semantic segmentation model:
    
    ```shell
    python run_segmentation.py
    ```

3) The results are written to the folder `output_monodepth` and `output_semseg`, respectively.

Use the flag `-t` to switch between different models. Possible options are `dpt_hybrid` (default) and `dpt_large`.


**Additional models:**

- Monodepth finetuned on KITTI: [dpt_hybrid_kitti-cb926ef4.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt) [Mirror](https://drive.google.com/file/d/1-oJpORoJEdxj4LTV-Pc17iB-smp-khcX/view?usp=sharing)
- Monodepth finetuned on NYUv2: [dpt_hybrid_nyu-2ce69ec7.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt) [Mirror](https\://drive.google.com/file/d/1NjiFw1Z9lUAfTPZu4uQ9gourVwvmd58O/view?usp=sharing)

Run with 

```shell
python run_monodepth -t [dpt_hybrid_kitti|dpt_hybrid_nyu] 
```

4) If you want to support new method on the project, just follow 4 steps

+ Create a new python file in `models/`, and code your structure to replace vision transformer.
+ Add your structure to the  method`_make_encoder` of `dpt/blocks.py`.
+ Add the hook pair in class `DPT` in `dpt/models.py`.
+ Add setting in `util/parser.py`.

Now, you can use your method on the framework of DPT.

### Evaluation

Hints on how to evaluate monodepth models can be found in [EVALUATION.md](./EVALUATION.md).


### Citation

Please cite our papers if you use this code or any of the models. 

```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```

```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

```
@article{liang2022expediting,
	author    = {Liang, Weicong and Yuan, Yuhui and Ding, Henghui and Luo, Xiao and Lin, Weihong and Jia, Ding and Zhang, Zheng and Zhang, Chao and Hu, Han},
	title     = {Expediting large-scale vision transformer for dense prediction without fine-tuning},
	journal   = {arXiv preprint arXiv:2210.01035},
	year      = {2022},
}
```

### Acknowledgements

Our work builds on and uses code from [timm](https://github.com/rwightman/pytorch-image-models) and [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). We'd like to thank the authors for making these libraries available.

### License 

MIT License 
