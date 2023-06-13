## Contains configurations for models and parameters

from yacs.config import CfgNode as CN
import torch

CFG = CN()
CFG.SEED = 11997733
CFG.MSRSYNC = False

# Plan ViT model parameters
CFG.MODEL = CN()
CFG.MODEL.PATCHSIZE = 16
CFG.MODEL.MIXUP=True
CFG.MODEL.VERSION=1 # grouped embedding
CFG.MODEL.SUBBLOCK=True # sub-block conversion enabled
CFG.MODEL.CLASSES=1000 # number of output classes
CFG.MODEL.DTYPE='fp32'
CFG.MODEL.AMPDTYPE='fp16'
CFG.MODEL.SUMCOL=['output_size', 'num_params'] # Summary columns for torchinfo

# Training pipeline hyperparameters
CFG.TRAIN = CN() 
CFG.TRAIN.EPOCHS=300
CFG.TRAIN.BATCHSIZE=1024 # batch size across all GPUs. (ex. (BATCHSIZE=1024, 4 GPUs): batch size per gpu = 256) # CFG.TRAIN.BATCHPERGPU will contain batch size per gpu after running update_config() function in pipeline_utils.py
CFG.TRAIN.LR=3e-3
CFG.TRAIN.WD=3e-4
CFG.TRAIN.DROP=0.0
CFG.TRAIN.WARMUP=10000
CFG.TRAIN.AUGLIST="AutoContrast,Posterize,Color,Contrast,Brightness,Sharpness,Cutout,TranslateX,TranslateY,Rotate90,AutoSaturation,Grayscale,MidfreqAug,ChromaDrop".split(",") # default augmentation list for DCT
CFG.TRAIN.NUMOPS=2
CFG.TRAIN.AUGSTR=3 # default augmentation strength (ops_magnitude)
CFG.TRAIN.AUGMAX=10 # maximum augmentation strength
CFG.TRAIN.SPLIT=0.01 # Split train data to use as minival
CFG.TRAIN.DETERMINISTIC=True # enables deterministic mode -- is slower but the results are (more or less) reproducible

# AMP related settings
CFG.TRAIN.SCALER = CN()
CFG.TRAIN.SCALER.GROWTH=1.6
CFG.TRAIN.SCALER.BACKOFF=0.625
CFG.TRAIN.SCALER.INTERVAL=600

# Placeholder settings
CFG.RANK=0
CFG.WORLDSIZE=1
CFG.THREADS=1
CFG.TRAIN.BATCHPERGPU=128
CFG.TRAIN.RUNTRAIN=True # Run train pipeline if true
CFG.TRAIN.RUNEVAL=True # Run eval pipeline if true
CFG.TRAIN.DATASET='imagenet'
CFG.TRAIN.DETERMINISTIC=False # Set deterministic mode if True
CFG.MODEL.DOMAIN="DCTorRGB"
CFG.MODEL.INPUTSIZE=(1,3,224,224)
CFG.MODEL.SUMDTYPE=['fp32']
CFG.MODEL.HEADS = 6
CFG.MODEL.HEADSIZE = 64
CFG.MODEL.EMBEDSIZE = 384
CFG.MODEL.DEPTH = 12
CFG.TRAIN.AMP=False

def generate_config(
    modelarch="vits", domain="dct", modelver=None, subblock=None,
    epochs=None, batchsize=None, lr=None, wd=None, drop=None, warmup_steps=None, auglist=None,
    num_ops=None, ops_magnitude=None, augstr=None, seed=None, amp=None, ampdtype=None, use_msrsync=None,
    ):
    """
    Generate appropriate config for given hyperparameters

    Inputs
        modelarch: Model architecture. (ex: vitti, vits, vitb, vitl, swinv2)

        (Optional) If set, override default settings.
        - modelver: 1: concat embedding, 2: grouped embedding, 3: separate embedding 
        - subblock(T/F): use sub-block conversion)
        - epochs, batchsize, lr, wd, drop, warmup_steps, batchsize, num_ops, augstr(0-10), seed, amp(T/F), use_msrsync(T/F),
        - ampdtype: string chosen from (fp32, fp16, bf16)
        - auglist: string separated by ','
    
    Outputs
        cfg: yacs config (case sensitive)
    """
    # todo: support variable patch size (some power of 2)
    cfg = CFG.clone()
    cfg.MODEL.DOMAIN = domain.upper()
    # Model params
    cfg.MODEL.ARCH=modelarch
    if modelarch=='vitti':
        cfg.MODEL.HEADS = 3
        cfg.MODEL.HEADSIZE = 64
        cfg.MODEL.EMBEDSIZE = 192
        cfg.MODEL.DEPTH = 12
        cfg.MODEL.PATCHSIZE=16
        if cfg.MODEL.DOMAIN=="DCT":
            cfg.TRAIN.AUGLIST="AutoContrast,Posterize,SolarizeAdd,Color,Contrast,Brightness,MidfreqAug,Cutout,TranslateX,TranslateY,Rotate90,AutoSaturation,Grayscale,ChromaDrop".split(",") # + SolarizeAdd, - Sharpness
        cfg.TRAIN.WD=1e-4
        cfg.TRAIN.AMP=False
    elif modelarch=='vits':
        cfg.MODEL.HEADS = 6
        cfg.MODEL.HEADSIZE = 64
        cfg.MODEL.EMBEDSIZE = 384
        cfg.MODEL.DEPTH = 12
        cfg.MODEL.PATCHSIZE=16
        cfg.TRAIN.EPOCHS=90
        cfg.TRAIN.AMP=False
    elif modelarch=='vitb':
        cfg.MODEL.HEADS = 12
        cfg.MODEL.HEADSIZE = 64
        cfg.MODEL.EMBEDSIZE = 768
        cfg.MODEL.DEPTH = 12
        cfg.MODEL.PATCHSIZE=16
        cfg.TRAIN.AMP=True
        cfg.MODEL.AMPDTYPE='bf16'
        cfg.TRAIN.LR=1e-3
        cfg.TRAIN.WD=1e-4
        cfg.TRAIN.BATCHSIZE=512
    elif modelarch=='vitl': # Not tested
        cfg.MODEL.HEADS = 12
        cfg.MODEL.HEADSIZE = 64
        cfg.MODEL.EMBEDSIZE = 1024
        cfg.MODEL.DEPTH = 24
        cfg.MODEL.PATCHSIZE=16
        cfg.TRAIN.AMP=True
        cfg.MODEL.AMPDTYPE='bf16'
    elif modelarch=='swinv2':
        cfg.MODEL.HEADS = [3,6,12,24]
        cfg.MODEL.EMBEDSIZE = 96
        cfg.MODEL.DEPTH = [2,2,6,2]
        cfg.MODEL.WINDOWSIZE = 8 # 8 or 16
        cfg.MODEL.MLPRATIO = 4
        cfg.MODEL.DROP = 0
        cfg.MODEL.DROPATTN = 0
        cfg.MODEL.DROPPATH = 0.2
        cfg.MODEL.QKVBIAS = True
        cfg.MODEL.APE = False
        cfg.MODEL.PNORM = True
        cfg.MODEL.PRETRAINED=[0,0,0,0]
        cfg.MODEL.PATCHSIZE=4
        cfg.TRAIN.AMP=True
        cfg.TRAIN.BATCHSIZE=512
    if modelver != None:
        cfg.MODEL.VERSION = modelver
    if subblock != None:
        cfg.MODEL.SUBBLOCK = subblock
    if epochs != None:
        cfg.TRAIN.EPOCHS = epochs
    if lr != None:
        cfg.TRAIN.LR=lr
    if wd != None:
        cfg.TRAIN.WD=wd
    if drop != None:
        cfg.TRAIN.DROP=drop
    if warmup_steps != None:
        cfg.TRAIN.WARMUP=warmup_steps
    if num_ops != None:
        cfg.TRAIN.NUMOPS = num_ops
    if ops_magnitude != None:
        cfg.TRAIN.AUGSTR = ops_magnitude
    if augstr != None:
        cfg.TRAIN.AUGMAX = augstr
    if seed != None:
        cfg.SEED = seed
    if batchsize != None:
        cfg.TRAIN.BATCHSIZE=batchsize
    if auglist != None:
        cfg.TRAIN.AUGLIST=auglist.split(",")
    if amp != None:
        cfg.TRAIN.AMP=bool(amp)
    if ampdtype != None:
        cfg.MODEL.AMPDTYPE=ampdtype
    if use_msrsync != None:
        cfg.MSRSYNC=use_msrsync

    if cfg.MODEL.DOMAIN=="RGB":
        cfg.TRAIN.LR=1e-3
        cfg.TRAIN.WD=1e-4
        cfg.TRAIN.AUGLIST="AutoContrast,Equalize,Contrast,Brightness,Color,Sharpness,Posterize,Invert,Solarize,SolarizeAdd,TranslateX,TranslateY,Cutout,Rotate,ShearX,ShearY".split(',')
        cfg.TRAIN.AUGSTR=10 # default augmentation strength (ops_magnitude)

    return cfg