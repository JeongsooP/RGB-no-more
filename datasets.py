# May 3 2022 11:40
# Jeongsoo Park

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import utils.custom_transforms as ctrans
from utils.custom_sampler import DistributedEvalSampler
import dct_manip as dm # custom DCT coefficient handler
from utils.mp_scripts import tar_extractor
import random
from functools import partial

import pandas as pd
import numpy as np
import os
import tarfile
import shutil
import sysrsync
import PIL.Image
from time import time
import logging

class UnsupportedDataset(Exception):
    def __init__(self, msg='Unsupported dataset exception', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class imagenet_tar_handler():
    def __init__(
        self, 
        tarpath:str="Path_above_tar", 
        tmppath:str="/tmp/tempdata/imagenet",
        tmpprefix:str="imgnet_data",
        tarnames=['ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'], 
        valprepname='valprep.sh',
        num_proc=4,
        verbose=True
        ):
        """
        tarpath: Path containing original tar files and valprep.sh file
        tmppath: Path to temporary dataset (parent to 'imgnet_data' and 'tarfiles')
        tmpprefix: Prefix for unpacked imagenet data. imagenet data will be unpacked to: <tmppath>/<tmpprefix>/<train or val>/*
        tarnames: [<train tar file name>, <val tar file name>]
        valprepname: .sh file that contains the information for processing val data. Can be downloaded from https://github.com/soumith/imagenet-multiGPU.torch -- moves validation images to appropriate subfolders.
        num_proc: number of processes to use when unpacking
        verbose: If True, prints progress
        """
        self.tarpath=tarpath
        self.tmppath=tmppath
        self.tmpprefix=tmpprefix
        self.originaldir = os.getcwd() # directory to go back to after operation
        self.tarnames=tarnames
        self.valprepname=valprepname
        self.num_proc=num_proc
        self.verbose=verbose

    def checkfile(self):
        """
        Method that checks if there are correct files in .tar path
        returns True if verified, False if not.
        """
        check_target = len(self.tarnames) + 1
        check_count = 0
        check_files = self.tarnames.copy()
        check_files.append(self.valprepname)
        for file in os.listdir(self.tarpath):
            if file in check_files:
                check_count += 1
        
        if check_count == check_target: 
            logging.info(f"All files verified: {check_files}")
            return True
        else:
            logging.info(f"Files not verified. Files in tarpath: {os.listdir(self.tarpath)}")
            return False

    def tar2tmp(self, copy=True, unpack=True, extract_train=True, extract_val=True):
        """
        Method that copies tar file to tmppath and optionally unpacks them.
        Uses rsync to show progress bar
        
        inputs
            copy: If true, copy tar files from source to destination
            unpack: If true, unpack .tar files after copying
            extract_train: If true, extract <train>.tar
            extract_val: If true, extract <val>.tar

        returns
            1: copying done (no unpacking)
            2: unpacking done
        """
        if copy:
            copy_starttime = time()
            if not os.path.isdir(os.path.join(self.tmppath, 'tarfile')):
                os.makedirs(os.path.join(self.tmppath, 'tarfile')) # /<TMP>/tarfile/
            for tarname in self.tarnames:
                tarfilepath = os.path.join(self.tarpath, tarname) # source (/<SRC>/tarfile.tar)
                tmpfilepath = os.path.join(
                    os.path.join(self.tmppath, 'tarfile')) # dest (/<TMP>/tarfile/)
                sysrsync.run(
                    source=tarfilepath, 
                    destination=tmpfilepath, 
                    options=['--progress']) # copy source to dest
                if self.verbose:
                    logging.info(f"File {tarname} copied to {self.tmppath}")
            copy_endtime=time()
            copytime = copy_endtime - copy_starttime
            if self.verbose:
                logging.info(f"Copying done in {copytime//60:2.0f}m {copytime%60:2.0f}s")


        if unpack:
            unpack_starttime=time()
            if self.tmpprefix != "":
                imgnet_trainpath = os.path.join(self.tmppath, f'{self.tmpprefix}/train') # /<TMP>/<tmpprefix>/train
                imgnet_valpath = os.path.join(self.tmppath, f'{self.tmpprefix}/val') # /<TMP>/<tmpprefix>/val
            else:
                imgnet_trainpath = os.path.join(self.tmppath, 'train') # /<TMP>/train
                imgnet_valpath = os.path.join(self.tmppath, 'val') # /<TMP>/val
            if not os.path.isdir(imgnet_trainpath):
                os.makedirs(imgnet_trainpath)
            if not os.path.isdir(imgnet_valpath):
                os.makedirs(imgnet_valpath)
            sysrsync.run(
                source=os.path.join(self.tarpath, self.valprepname),
                destination=imgnet_valpath, 
                options=['--progress']
            ) # copy valprep.sh to imgnet_valpath
            if self.verbose:
                logging.info(f"File {self.valprepname} copied to {imgnet_valpath}")

            if extract_train:
                if copy:
                    train_tar = tarfile.open(os.path.join(self.tmppath, f'tarfile/{self.tarnames[0]}')) # train data handling
                else:
                    train_tar = tarfile.open(os.path.join(self.tarpath, self.tarnames[0])) # train data handling
                if self.verbose:
                    logging.info(f"Extracting {os.path.join(self.tmppath, f'tarfile/{self.tarnames[0]}')} -- can take up to 1hr")
                train_tar.extractall(path=imgnet_trainpath)
            logging.info(f"Train tar files extracted. Now extracting individual tar files...")
            tarfilelist = [] # list of tar files
            for _, _, files in os.walk(imgnet_trainpath): # extract child .tar files
                for filename in files:
                    if filename[-4:].lower()=='.tar':
                        tarfilelist.append(filename)
            tarextract = tar_extractor(self.num_proc, tarfilelist, imgnet_trainpath, self.verbose)
            tarextract.start_processes()
            tarextract.wait_and_terminate()
            logging.info(f"Extracting training data done.")
            if extract_val:
                if copy:
                    val_tar = tarfile.open(os.path.join(self.tmppath, f'tarfile/{self.tarnames[1]}')) # val data handling
                else:
                    val_tar = tarfile.open(os.path.join(self.tarpath, self.tarnames[1])) # val data handling
                if self.verbose:
                    logging.info(f"Extracting {os.path.join(self.tmppath, f'tarfile/{self.tarnames[1]}')} -- can take up to 1hr")
                val_tar.extractall(path=imgnet_valpath)
            os.chdir(imgnet_valpath) # move to imgnet valpath directory
            os.system(f"sh {self.valprepname}") # run shell script
            os.remove(self.valprepname)
            os.chdir(self.originaldir) # go back to original directory
            logging.info(f"Extracting validation data done.")
            unpack_endtime=time()
            unpacktime = unpack_endtime - unpack_starttime
            if self.verbose:
                logging.info(f"Unpacking done in {unpacktime//60:2.0f}m {unpacktime%60:2.0f}")
            return 2
        else:
            return 1
    
    def remove_data(self):
        logging.warning("=== WARNING: Deleting path: {} ===".format(self.tmppath))
        shutil.rmtree(self.tmppath)
        logging.info("=== Path: {} deleted ===".format(self.tmppath))
        
def run_msrsync(
    msrsync_path='./msrsync',
    source='source_folder',
    dest='destination_folder',
    bucketpath="./msrsync_temp",
    process=1,
    verbose=True,
    ):
    """
    Runs msrsync from shell
    msrsync_path: path to msrsync executable
    source: source folder
    dest: destination folder (source folder gets copied as a child folder into the destination folder)
    bucketpath: path to msrsync bucket
    process: number of processes ro run
    verbose: if true, shows progress
    """
    if not os.path.isdir(bucketpath):
        os.makedirs(bucketpath)
    bucketpath = f" -b {bucketpath}"
        
    show_progress = " -P" if verbose else ""

    command = f"{msrsync_path} -p {process}{show_progress}{bucketpath} {source} {dest}"
    if not os.path.isdir(dest):
        os.makedirs(dest) # make directory to destination
    if verbose:
        print(f"[msrsync] copying {source} to {dest}")
    os.system(command) # runs command

def transcode_to_jpeg(
    filepath='something.jpeg',
):
    """
    Transcodes non-jpeg image to jpeg

    Args:
        filepath: path to not-jpeg file to transcode
    """
    notjpeg_image = PIL.Image.open(filepath).convert("RGB")
    filepath_basename = os.path.basename(filepath).split('.')[0] # filename without extension
    filepath_basename = filepath_basename + ".JPEG" # make it jpeg
    savefilename = os.path.join(
        os.path.dirname(filepath), filepath_basename
    )
    os.remove(filepath) # delete original file
    notjpeg_image.save(savefilename, 'jpeg') # write to jpeg


class imagenet_dataset_indexing(torch.utils.data.Dataset):
    """
    ImageNet folder structure should be: 
        /<turbo_path>/shared_datasets/imagenet
            - train
                - n01440764
                    - n01440764_10026.JPEG
                    - ...
                - ...
            - test
                - n01440764
                    - ILSVRC2012_val_00000293.JPEG
                    - ...
                - ...
    ImageNet label mapping from: https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57 by aaronpolhamus
    
    Imagenet Dataloader
    inputs
    - indexfile: .csv file that contains the indexing for imagenet dataset (format: [<filepath>, <label>])
    - labelmapfile: .txt file that maps labelname to label (txt line format: 'n02119789 1 kit_fox')
    - type: 'train' or 'test' for transform selection
    - baseindex: If True, use base index format (<path_to_trainval>/train/<class>/img.jpg). Otherwise, use raw index format (<path>/img.jpg)
    - load_mode: 'RGB' or 'DCT'. If RGB, decode jpeg into RGB. If DCT, use dct_manip library to decode until DCT coeff.
    - basepath: path containing train/val for imagenet data
    - dtype: data type
    - rank: GPU rank (== gpu device index)

    outputs
    - torch.utils.data.Dataset object for ImageNet
    """
    def __init__(
        self, 
        indexfile='path_to_indexfile.csv', 
        type='train', 
        basepath='', 
        load_mode='RGB',
        dtype=torch.float32,
        rank=-1):

        super(imagenet_dataset_indexing).__init__()
        self.imageindex = pd.read_csv(indexfile).to_numpy() # format: [<filepath>, <label>]
        self.type = type
        self.dtype = dtype
        self.basepath = basepath
        self.load_mode = load_mode
        self.transform = None # this will be handeled separately at SubsetWithTransform() class or get_transform() function
        self.rank = rank

    def __getitem__(self, index):
        # path_to_img = self.imageindex[index, 0] # not part of an actual code (can be though) -- just here to help understand what's going on
        # label = self.imageindex[index, 1]
        if self.basepath != '':
            imgpath = os.path.join(self.basepath, self.imageindex[index, 0])
        else:
            imgpath = self.imageindex[index, 0]
        if self.load_mode=='RGB' or self.load_mode=="RGB2DCT":
            img = PIL.Image.open(imgpath).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.imageindex[index,1] # image, label
        elif self.load_mode=='DCT' or self.load_mode=="DCT2RGB":
            dim, quant, Y, cbcr = dm.read_coefficients(imgpath) # dimension, quantization, Y, cbcr coefficients
            Y = torch.clamp(Y * quant[0], min=-2**10, max=2**10-8) # recover quantized coefficients (clamp to -1024~1023 (values should be within -1024~1016 with quant table of all ones))
            if cbcr is not None: # if colored
                cbcr = torch.clamp(cbcr * quant[1:3].unsqueeze(1).unsqueeze(1), min=-2**10, max=2**10-8) # recover quantized coefficient
            else: # if black and white
                _, h, w, kh, kw = Y.shape
                cbcr = torch.zeros((2,h//2,w//2,kh,kw), dtype=Y.dtype, device=Y.device) # fill it with zeroes
            coeffs = (Y, cbcr)
            if self.transform:
                coeffs = self.transform(coeffs)
            return coeffs, self.imageindex[index,1] # label
        elif self.load_mode=='GPU':
            img = torchvision.io.read_file(imgpath)
            return img, self.imageindex[index,1], len(img) # image, label, length (1D)

    def __len__(self):
        return len(self.imageindex)

def get_transform(dataset='imagenet', type='train', ops_list=None, 
    num_ops=2, ops_magnitude=10, 
    dtype=torch.float32, dtype_resize=torch.float32):
    """
    Returns appropriate transform for dataset type (Train/val/test)

    ops_list: list of operation names to use during randaugment (if None, use default)
    num_ops: number of operations to apply
    ops_magnitude: RandAugment magnitude
    dtype: dtype of the final data
    dtype_resize: dtype to use during resizing (for dct)
    """
    if dataset=='imagenet':
        if type=='train':
            transform = transforms.Compose([
                                transforms.RandomResizedCrop(224, scale=(0.05,1.0)),
                                transforms.RandomHorizontalFlip(),
                                ctrans.RandAugment_bv(num_ops=num_ops, magnitude=ops_magnitude, num_magnitude_bins=11, fill=128, ops_list=ops_list),
                                ctrans.ToTensor_range(val_min=-1, val_max=1),
                                transforms.ConvertImageDtype(dtype),
                                ])
        elif type=='val' or type=='test':
            transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            ctrans.ToTensor_range(val_min=-1, val_max=1),
                            transforms.ConvertImageDtype(dtype),
                            ])
        else:
            print("Unrecognized dataset type! Returning 'None' transform")
            transform = None
    elif dataset=='imagenet_swin': 
        if type=='train':
            transform = transforms.Compose([
                                transforms.RandomResizedCrop(256, scale=(0.05,1.0)),
                                transforms.RandomHorizontalFlip(),
                                ctrans.RandAugment_bv(num_ops=num_ops, magnitude=ops_magnitude, num_magnitude_bins=11, fill=128, ops_list=ops_list),
                                ctrans.ToTensor_range(val_min=-1, val_max=1),
                                transforms.ConvertImageDtype(dtype),
                                ])
        elif type=='val' or type=='test':
            transform = transforms.Compose([
                            transforms.Resize(256),
                            ctrans.ToTensor_range(val_min=-1, val_max=1),
                            transforms.ConvertImageDtype(dtype),
                            ])
        else:
            print("Unrecognized dataset type! Returning 'None' transform")
            transform = None
    elif dataset=='imagenet_dct':
        if type=='train':
            transform = transforms.Compose([
                                ctrans.RandomResizedCrop_DCT(28, scale=(0.05, 1.0), ratio=(1,1), dtype_resize=dtype_resize), # RR Crops
                                ctrans.RandomFlip_DCT(p=0.5, direction='horizontal'),
                                ctrans.RandAugment_dct(num_ops=num_ops, magnitude=ops_magnitude, num_magnitude_bins=11, ops_list=ops_list),
                                ctrans.ToRange(val_min=-1, val_max=1, orig_min=-1024, orig_max=1016, dtype=dtype),
                                ])
        elif type=='val' or type=='test':
            transform = transforms.Compose([
                            ctrans.ResizedCenterCrop_DCT(32, 28, dtype_resize=dtype_resize), # equivalent to resize 32 -> center crop 28. This instead crops 56 -> resize 28 for faster resizing
                            ctrans.ToRange(val_min=-1, val_max=1, orig_min=-1024, orig_max=1016, dtype=dtype),
                            ])
        else:
            print("Unrecognized dataset type! Returning 'None' transform")
            transform = None
    elif dataset=='imagenet_dct_swin':
        if type=='train':
            transform = transforms.Compose([
                                ctrans.RandomResizedCrop_DCT(32, scale=(0.05, 1.0), ratio=(1,1), dtype_resize=dtype_resize), # RR Crops
                                ctrans.RandomFlip_DCT(p=0.5, direction='horizontal'),
                                ctrans.RandAugment_dct(num_ops=num_ops, magnitude=ops_magnitude, num_magnitude_bins=11, ops_list=ops_list),
                                ctrans.ToRange(val_min=-1, val_max=1, orig_min=-1024, orig_max=1016, dtype=dtype),
                                ])
        elif type=='val' or type=='test':
            transform = transforms.Compose([
                            ctrans.Resize_DCT(32),
                            ctrans.ToRange(val_min=-1, val_max=1, orig_min=-1024, orig_max=1016, dtype=dtype),
                            ])
        else:
            print("Unrecognized dataset type! Returning 'None' transform")
            transform = None
    else:
        print("Unrecognized dataset! Returning 'None' transform")
        transform = None
        
    return transform

class SubsetWithTransform(torch.utils.data.Dataset):
    """
    Custom subset class which allows to customize transform for each subsets got from random_split()
    """
    def __init__(self, subset, dataset=None, transform=None):
        self.subset = subset
        self.transform = transform
        self.dataset_name = dataset
    
    def __getitem__(self, index):
        img, lab = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, lab
    
    def __len__(self):
        return len(self.subset)

def set_seeds_for_data(seed=11997733):
    """
    Set seeds for Python, numpy, and pytorch. Used to split the dataset consistantly across DDP instances.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_seeds_for_worker(seed=11997733, id=0):
    """
    Set seeds for python, and numpy. Default seed=11997733.
    PyTorch seeding is handled by torch.Generator passed into the DataLoader
    """
    seed = seed % (2**31)
    random.seed(seed+id)
    np.random.seed(seed+id)

def worker_seed_reporter(id=None):
    """
    Reports worker seeds
    """
    workerseed = torch.utils.data.get_worker_info().seed
    numwkr = torch.utils.data.get_worker_info().num_workers
    baseseed = torch.initial_seed()
    logging.info(f"Worker id: {id}/{numwkr}, worker seed: {workerseed}, baseseed: {baseseed}, workerseed % 2**31: {workerseed % (2**31)}")

def set_seeds_and_report(seed=11997733, report=False, id=0):
    """
    Debug: set seeds and report
    """
    workerseed = torch.utils.data.get_worker_info().seed
    set_seeds_for_worker(workerseed, id)
    if report:
        worker_seed_reporter(id)

def dataset_selector(
    dataset='imagenet_dct', 
    type='train',
    indexpath='', 
    basepath='',

    batch_size=128, 
    num_workers=4, 
    shuffle=False, 
    trainval_split=-1,
    return_indices=False,

    distributed=False,
    rank=-1,
    world_size=-1,
    seed=None,
    
    ops_list=None,
    num_ops=2,
    ops_magnitude=10,
    dtype=torch.float32
    ):
    """
        Selects dataset and returns dataloader
        input
        - dataset: selects dataset
        - type: 'train' or 'test'
        - indexpath: path to indexfile
        - basepath: path containing train/val for imagenet data

        - batch_size: size of batch
        - num_workers: number of subprocesses to spawn
        - shuffle: shuffle dataset or not
        - trainval_split: split dataset if specified.
        - return_indices: if True, returns the split index (for reproducing results if seeding isn't consistant across machines/versions)

        - distributed: flag to whether it should return distributed dataloader or not (True if using Distributed Data Parallel)
        - rank: index of the current distributed process (index of a gpu)
        - world_size: size of the process group (# of gpus)
        - seed: set seeds to this value if not None

        - ops_list: list of operation name strings to use during randaugment (currently only imagenet_dct_256 dataset use this)
        - num_ops: number of operations to apply
        - ops_magnitude: magnitude of randaugment (default:10 / max 10)
        - dtype: dtype of dataset

        output
        - (Optional) Tuple of ('trainset', 'valset')
        - 'testset'
    """
    if dataset=='imagenet' or dataset=="imagenet_swin":
        dataset_object = imagenet_dataset_indexing(
            indexfile=indexpath, type=type,
            basepath=basepath, load_mode='RGB', dtype=dtype,
            )
    elif dataset[0:12]=='imagenet_dct':
        dataset_object = imagenet_dataset_indexing(
            indexfile=indexpath, type=type,
            basepath=basepath, load_mode='DCT', dtype=dtype,
            )
    else:
        raise UnsupportedDataset(msg="Unsupported dataset: {}".format(dataset))
    if seed is not None:
        seedftn = partial(set_seeds_and_report, seed+rank, False) #report=False
        seed_generator = torch.Generator(device='cpu')
        seed_generator.manual_seed(seed+rank)
    else:
        seedftn = None
    if trainval_split > 0: # train/val split
        set_seeds_for_data(seed) # Set same seeds for dataset split

        dataset_len = len(dataset_object) # whole dataset length
        vallen = int(trainval_split * dataset_len) # validation length
        trainlen = dataset_len - vallen # train length
        traindata_split, valdata_split = torch.utils.data.random_split(dataset_object, (trainlen, vallen))
        trainvaldata_split = torch.utils.data.Subset(traindata_split, range(0,round(trainlen*0.05))) # also evaluate model performance on 5% train data

        set_seeds_for_data(seed+rank) # after dataset is split, use different seeds for augmentations and shuffling.

        traindata_subset = SubsetWithTransform(traindata_split, dataset=dataset, transform=get_transform(dataset=dataset, type='train', ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude, dtype=dtype))
        valdata_subset = SubsetWithTransform(valdata_split, dataset=dataset, transform=get_transform(dataset=dataset, type='val', dtype=dtype))
        trainvaldata_subset = SubsetWithTransform(trainvaldata_split, dataset=dataset, transform=get_transform(dataset=dataset, type='val', dtype=dtype))
        train_subset_index = traindata_split.indices
        val_subset_index = valdata_split.indices

        train_sampler=None
        val_sampler=None
        if distributed:
            train_sampler = DistributedSampler(
                traindata_subset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False,
                )
            val_sampler = DistributedEvalSampler(
                valdata_subset, num_replicas=world_size, rank=rank, shuffle=False,
                ) # don't distribute val
            trainval_sampler = DistributedEvalSampler(
                trainvaldata_subset, num_replicas=world_size, rank=rank, shuffle=False
            )
        trainloader_obj = torch.utils.data.DataLoader(traindata_subset, batch_size=batch_size, pin_memory=True,
                                                shuffle=False if distributed else shuffle, num_workers=num_workers, 
                                                sampler=train_sampler, prefetch_factor=8 if not num_workers==0 else 2, 
                                                persistent_workers=True if not num_workers==0 else False,
                                                collate_fn = None, worker_init_fn=seedftn, generator=seed_generator)
        valloader_obj = torch.utils.data.DataLoader(valdata_subset, batch_size=batch_size, pin_memory=True,
                                                shuffle=False, num_workers=num_workers, sampler=val_sampler,
                                                prefetch_factor=8 if not num_workers==0 else 2, 
                                                persistent_workers=True if not num_workers==0 else False,
                                                collate_fn = None, worker_init_fn=seedftn, generator=seed_generator)
        trainvalloader_obj = torch.utils.data.DataLoader(trainvaldata_subset, batch_size=batch_size, pin_memory=True,
                                                shuffle=False, num_workers=num_workers, sampler=trainval_sampler,
                                                prefetch_factor=8 if not num_workers==0 else 2, 
                                                persistent_workers=True if not num_workers==0 else False,
                                                collate_fn = None, worker_init_fn=seedftn, generator=seed_generator)

        if rank==0:
            logging.info("Train/Val split: num_total: {}, num_train: {}, num_val: {}, num_trainval: {}".format(dataset_len, trainlen, vallen, round(trainlen*0.05)))

        if not return_indices:
            return trainloader_obj, valloader_obj, trainvalloader_obj
        else:
            return trainloader_obj, valloader_obj, trainvalloader_obj, train_subset_index, val_subset_index
    else:
        sampler=None
        datasubset = SubsetWithTransform(dataset_object, dataset=dataset, transform=get_transform(dataset=dataset, type=type, ops_list=ops_list if type=='train' else None, num_ops=num_ops, dtype=dtype))
        if distributed:
            if type=='train':
                sampler = DistributedSampler(
                    datasubset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False,
                    )
            else:
                sampler = DistributedEvalSampler(
                    datasubset, num_replicas=world_size, rank=rank, shuffle=shuffle,
                )
        dataloader_obj = torch.utils.data.DataLoader(datasubset, batch_size=batch_size, pin_memory=True,
                                                shuffle=False if distributed else shuffle, num_workers=num_workers, 
                                                sampler=sampler, prefetch_factor=8 if not num_workers==0 else 2, 
                                                persistent_workers=True if not num_workers==0 else False,
                                                collate_fn = None, worker_init_fn=seedftn, generator=seed_generator)
        return dataloader_obj