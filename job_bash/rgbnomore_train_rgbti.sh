python train.py --model_arch=vitti --domain=rgb --datapath=/nfs/imagenet/tarfiles --temp_datapath=/tmp/jespark/tempdata/imagenet --indexpaths=assets/indexbase_train.csv,assets/indexbase_val.csv --num_gpus=8 --num_cpus=32 --train --eval --savepath=/nfs/ViT/rgbnomore_rgbti.pth --verbose=1

