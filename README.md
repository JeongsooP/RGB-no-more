# RGB no more: Minimally Decoded JPEG Vision Transformers

An official code release of the paper [RGB no more: Minimally Decoded JPEG Vision Transformers](https://openaccess.thecvf.com/content/CVPR2023/html/Park_RGB_No_More_Minimally-Decoded_JPEG_Vision_Transformers_CVPR_2023_paper.html)\
This repository contains the pipeline for training JPEG ViT, including DCT augmentations.

If you use this repo, please cite our paper:

BibTex:
```plain
@InProceedings{Park_2023_CVPR,
    author    = {Park, Jeongsoo and Johnson, Justin},
    title     = {RGB No More: Minimally-Decoded JPEG Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22334-22346}
}
```

Plain text:
<blockquote>
Jeongsoo Park, and Justin Johnson. "RGB no more: Minimally-decoded JPEG Vision Transformers." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22334-22346. 2023.
</blockquote>

This readme heavily borrows the format by [SwinV2](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md).

## Pretrained Models

Pretrained models are available as:

| Model  | Val Acc (%) | Link |
| --- | --- | --- |
| JPEG-Ti  | 75.1 | [link](http://www-personal.umich.edu/~jespark/rgbnomore-2023/imgnetDCTViTTi_ep300_75.1.pth) | 
| ViT-Ti  | 74.1 | [link](http://www-personal.umich.edu/~jespark/rgbnomore-2023/imgnetRGBViTTi_ep300_74.1.pth) | 
| JPEG-S | 76.5 | [link](http://www-personal.umich.edu/~jespark/rgbnomore-2023/imgnetDCTViTS_ep90_76.5.pth) |
| ViT-S | 76.6 | [link](http://www-personal.umich.edu/~jespark/rgbnomore-2023/imgnetRGBViTS_ep90_76.5.pth) |
| SwinV2-T <br>(DCT, window=8) | 79.4 | [link](http://www-personal.umich.edu/~jespark/rgbnomore-2023/imgnetSwinDCT_ep300_79.4.pth) |
| SwinV2-T <br>(RGB, window=8) | 79.0 | [link](http://www-personal.umich.edu/~jespark/rgbnomore-2023/imgnetSwinRGB_ep300_79.0.pth) |

It is possible to download using `wget` using `wget [link]`. One example:

`wget http://www-personal.umich.edu/~jespark/rgbnomore-2023/imgnetDCTViTTi_ep300_75.1.pth`

Note: RGB ViT-S is trained on raw ImageNet to reproduce the [recipe by Google](https://arxiv.org/abs/2205.01580). Others are trained on 512x512 resized ImageNet.

## Usage

### Install

*Note that modified `libjpeg` requires **Linux** to compile properly. Other OS are not supported.*

- Clone this repo:

```bash
git clone https://github.com/JeongsooP/RGB-no-more.git
cd RGB-no-more
```

- Create a conda virtual environment and activate it:

```bash
conda create -n rgbnomore python=3.10
conda activate rgbnomore
```

- Install `PyTorch>=1.12.1` `CUDA>=11.3` (we used `PyTorch=1.12.1` and `CUDA=11.3` in our paper):
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 "llvm-openmp<16.0" -c pytorch
```

*We recommend using `llvm-openmp<16.0` until the [multiprocessing issue](https://github.com/pytorch/pytorch/issues/101850) is fixed.*

- Install `gcc_linux-64==12.2.0`, `gxx_linux-64==12.2.0` and other necessary libraries:
```bash
conda install -c conda-forge gcc_linux-64==12.2.0 gxx_linux-64==12.2.0 torchmetrics torchinfo tensorboard einops scipy yacs pandas timm imageio iopath
```

*Make sure the versions for `gcc_linux-64` and `gxx_linux-64` are **exactly 12.2.0.***

- Install other requirements:

```
pip install -r requirements.txt
```

- Compile `dct_manip` -- a modified `libjpeg` handler:

  - Open `dct_manip/setup.py`
  - Modify `include_dirs` and `library_dirs` to include your include and library folder.
  - Modify `extra_objects` to the path containing `libjpeg.so`
  - Modify `headers` to the path containing `jpeglib.h`

  - Run `cd dct_manip`
  - Run `pip install .`


### Data preparation

- Download the ImageNet dataset from http://image-net.org/.
- Place the `tar` files and `valprep.sh` in the same folder.
- The dataloader will automatically organize the data as following:
  ```bash
  imgnet_data
  ├── train
  │   ├── n01440764
  │   │   ├── n01440764_10026.jpeg
  │   │   ├── n01440764_10027.jpeg
  │   │   └── ...
  │   ├── n01443537
  │   │   ├── n01443537_10007.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── n01440764
      │   ├── ILSVRC2012_val_00000293.jpeg
      │   ├── ILSVRC2012_val_00002138.jpeg
      │   └── ...
      ├── n01443537
      │   ├── ILSVRC2012_val_00000236.jpeg
      │   └── ...
      └── ...
  ```
Images will be resized to 512x512 unless `--no_resize` flag is used.

### Pipeline

<details>
  <summary>Arguments (click to expand)</summary>
  
  ```
  usage: train.py [-h] [--port PORT] [--model_arch MODEL_ARCH] [--no_subblock] [--embed_type EMBED_TYPE] [--domain DOMAIN] [--datapath DATAPATH] [--temp_datapath TEMP_DATAPATH] [--indexpaths INDEXPATHS] [--delete_dataset] [--no_extract] [--no_resize] [--num_gpus NUM_GPUS] [--num_cpus NUM_CPUS] [--train] [--eval] [--benchmark BENCHMARK] [--savepath SAVEPATH]
                [--loadpath LOADPATH] [--load_ckpt LOAD_CKPT] [--deterministic] [--verbose VERBOSE] [--epochs EPOCHS] [--batch BATCH] [--lr LR] [--wd WD] [--drop DROP] [--warmup_steps WARMUP_STEPS] [--ops_list OPS_LIST] [--num_ops NUM_OPS] [--ops_magnitude OPS_MAGNITUDE] [--amp AMP] [--ampdtype AMPDTYPE] [--seed SEED] [--use_msrsync]

options:
  -h, --help            show this help message and exit
  --port PORT           Port for pytorch distributed dataparallel
  --model_arch MODEL_ARCH
                        Model architecture (vitti, vits, vitb, vitl, swinv2)
  --no_subblock         If set, disable subblock conversion
  --embed_type EMBED_TYPE
                        Embedding layer type. (1: grouped, 2: separate, 3: concatenate). Default 1
  --domain DOMAIN       (DCT/RGB) Choose domain type
  --datapath DATAPATH   Path to folder containing the .tar files
  --temp_datapath TEMP_DATAPATH
                        Path to extract .tar files
  --indexpaths INDEXPATHS
                        Path to train/val index files. Separated by comma.
  --delete_dataset      If set, delete dataset after train or eval
  --no_extract          If set, assume .tar is already extracted to temp_datapath
  --no_resize           If set, do not resize images in temp_datapath to 512x512
  --num_gpus NUM_GPUS   number of GPUs to use. If not set, automatically use all available GPUs
  --num_cpus NUM_CPUS   number of total available cpu threads
  --train               Train new model
  --eval                Evaluate model loaded from ``savepath``
  --benchmark BENCHMARK
                        If set, benchmark for the set iterations
  --savepath SAVEPATH   Save path for model. Also saves checkpoint at this path
  --loadpath LOADPATH   Load path for model. Used during evaluation. If empty, copy savepath
  --load_ckpt LOAD_CKPT
                        If set, load checkpoint from this path
  --deterministic       If set, use deterministic mode
  --verbose VERBOSE     (0/1/2) 0: no output, 1: output per epoch, 2: output per iteration
  --epochs EPOCHS       Override the number of epochs
  --batch BATCH         Override the size of batch (overall batch size)
  --lr LR               Override the learning rate
  --wd WD               Override the weight decay strength
  --drop DROP           Override dropout probability
  --warmup_steps WARMUP_STEPS
                        Override warmup steps
  --ops_list OPS_LIST   Override augmentation list
  --num_ops NUM_OPS     Override number of operations
  --ops_magnitude OPS_MAGNITUDE
                        Override augmentation magnitude
  --amp AMP             (True:1/False:0) Override automatic mixed precision
  --ampdtype AMPDTYPE   Override amp dtype casting
  --seed SEED           Override random seed
  --use_msrsync         If set, use msrsync instead of .tar
  ```
  
  </details>

Sample bash script for all models are included in [./job_bash](./job_bash). Make sure to modify `--datapath`, `--temp_datapath`, `--num_gpus`, `--num_cpus`, `--savepath`, `--loadpath`, `--verbose` and any other flags to your setting.

- #### Training from scratch on ImageNet-1K

To train a `JPEG-Ti` on ImageNet from scratch, run:

<blockquote>
python train.py --model_arch=vitti --embed_type=1 --domain=dct --datapath=/path_to/tarfiles --temp_datapath=/tmp/temp_path --indexpaths=assets/indexbase_train.csv,assets/indexbase_val.csv --num_cpus=16 --savepath=/path_to_save/jpegti_model.pth --verbose=2  --train --eval
</blockquote>

- #### Evaluation

To evaluate a pre-trained `JPEG-Ti` on ImageNet val, run:

> python train.py --model_arch=vitti --embed_type=1 --domain=dct --datapath=/path_to/tarfiles --temp_datapath=/tmp/temp_path --indexpaths=assets/indexbase_train.csv,assets/indexbase_val.csv --num_cpus=16 --loadpath=/model_savedpath/jpegti_model.pth --verbose=2 --eval

You can change the `--model_arch` option to the corresponding model architecture (vitti, vits, vitb, swinv2).

- #### Throughput

To measure the throughput for 128 iterations, run:

> python train.py --model_arch=vitti --embed_type=1 --domain=dct --datapath=/path_to/tarfiles --temp_datapath=/tmp/temp_path --indexpaths=assets/indexbase_train.csv,assets/indexbase_val.csv --num_cpus=16 --loadpath=/model_savedpath/jpegti_model.pth --verbose=2 --benchmark 128
