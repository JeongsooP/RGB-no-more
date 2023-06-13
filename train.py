# Pytorch
import torch
import torch.distributed as dist
import torch.multiprocessing as torchmp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import logging
import shutil
import utils.configs as configs
import eval as eval
import utils.pipeline_utils as utils

def parse_args():
    """
    Parse arguments
    """
    parser=argparse.ArgumentParser()

    # DDP config
    parser.add_argument('--port', type=int, default=13932, help='Port for pytorch distributed dataparallel')

    # model config
    parser.add_argument('--model_arch', type=str, default='vits', help='Model architecture (vitti, vits, vitb, vitl, swinv2)')
    parser.add_argument('--no_subblock', action='store_true', help='If set, disable subblock conversion')
    parser.add_argument("--embed_type", type=int, default=2, help='Embedding layer type. (1: grouped, 2: separate, 3: concatenate). Default 1')
    parser.add_argument("--domain", type=str, default="dct", help="(DCT/RGB) Choose domain type")

    # data config
    parser.add_argument("--datapath", type=str, default='./imagenet', help='Path to folder containing the .tar files')
    parser.add_argument("--temp_datapath", type=str, default='/tmp/imagenet_data', help='Path to extract .tar files')
    parser.add_argument("--indexpaths", type=str, default='assets/index_train.csv,assets/index_val.csv', help='Path to train/val index files. Separated by comma.')
    parser.add_argument('--delete_dataset', action='store_true', help='If set, delete dataset after train or eval')
    parser.add_argument('--no_extract', action='store_true', help='If set, assume .tar is already extracted to temp_datapath')
    parser.add_argument('--no_resize', action='store_true', help='If set, do not resize images in temp_datapath to 512x512')
    parser.add_argument('--num_gpus', type=int, default=-1, help='number of GPUs to use. If not set, automatically use all available GPUs')
    parser.add_argument('--num_cpus', type=int, default=1, help='number of total available cpu threads')

    # pipeline config
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--eval', action='store_true', help='Evaluate model loaded from ``savepath`` ')
    parser.add_argument('--benchmark', type=int, default=0, help='If set, benchmark for the set iterations')
    parser.add_argument('--savepath', type=str, default='./models/ViT_v1.pth', help='Save path for model. Also saves checkpoint at this path')
    parser.add_argument('--loadpath', type=str, default='', help='Load path for model. Used during evaluation. If empty, copy savepath')
    parser.add_argument('--load_ckpt', type=str, default='', help='If set, load checkpoint from this path')
    parser.add_argument('--deterministic', action='store_true', help='If set, use deterministic mode')
    parser.add_argument('--verbose', type=int, default=0, help='(0/1/2) 0: no output, 1: output per epoch, 2: output per iteration')

    # override default config
    parser.add_argument("--epochs", type=int, default=-1, help="Override the number of epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Override the size of batch (overall batch size)")
    parser.add_argument("--lr", type=float, default=-1, help='Override the learning rate')
    parser.add_argument("--wd", type=float, default=-1, help='Override the weight decay strength')
    parser.add_argument('--drop', type=float, default=-1, help='Override dropout probability')
    parser.add_argument('--warmup_steps', type=int, default=-1, help='Override warmup steps')
    parser.add_argument('--ops_list', type=str, default='', help='Override augmentation list')
    parser.add_argument('--num_ops', type=int, default=-1, help='Override number of operations')
    parser.add_argument('--ops_magnitude', type=int, default=-1, help='Override augmentation magnitude')
    parser.add_argument("--amp", type=int, default=-1, help="(True:1/False:0) Override automatic mixed precision")
    parser.add_argument("--ampdtype", type=str, default='', help="Override amp dtype casting")
    parser.add_argument('--seed', type=int, default=-1, help='Override random seed')
    parser.add_argument('--use_msrsync', action='store_true', help='If set, use msrsync instead of .tar')

    args=parser.parse_args()    
    return args

def traineval(
    rank, # rank of GPU (i.e. gpu index)
    world_size,
    cfg,
    
    temp_datapath='./path_to_tmp',
    indexpaths=['path1.csv', 'path2.csv'],
    delete_dataset=False,
    #no_extract=False,
    num_cpus=4,

    run_train=False,
    run_eval=False,
    benchmark=0,
    savepath='path_to_model',
    loadpath='',
    load_ckpt='',
    deterministic=False,
    verbose=2,
    port=13932,
):
    """
    Trains and evaluates the ViT model

    rank: rank of GPU (gpu index)
    world_size: total number of GPUs
    
    cfg: yacs config node

    # model_type: type of model (vitti, vits, vitb, vitl, swinv2)
    # embed_type: type of embedding (1: concatenate, 2: grouped, 3: separate)
    # no_subblock: If true, do not use subblock conversion (default: False) ## included in config

    temp_datapath: path to extract .tar files
    indexpaths: list containing index file paths. [0] = train, [1] = val
    delete_dataset: If true, delete temp_datapath after pipeline is complete
    #no_extract: If true, assume the dataset is already extracted to temp_datapath
    num_cpus: Total number of CPUs available for this pipeline. Should be some integer multiple of number of GPUs

    run_train: If true, train new model
    run_eval: If true, evaluate model loaded from 'savepath'
    benchmark: If >0, run benchmark for the set iterations
    savepath: Path to save new model and checkpoints
    loadpath: Path to load models from
    load_ckpt: If set, load checkpoint from this path
    deterministic: If set, use deterministic algorithms (is slower)
    verbose: 0: no output, 1: output per epoch, 2: output per iteration
    port: Port to use for pytorch distributed dataparllel
    """
    utils.dist_setup(rank, world_size, port=port) # change port if needed
    if rank==0:
        writer = utils.get_tensorboard(cfg, savepath) # initialize tensorboard writer
    # initialize basic parameters
    utils.set_seeds(cfg.SEED+rank)
    threads = utils.get_threads(num_cpus, world_size, rank)
    cfg = utils.update_config(cfg, rank, world_size, threads, run_train, run_eval, deterministic)
    mixup = utils.get_mixup(cfg)
    # initialize data
    path_to_delete = utils.get_deletion_path(delete_dataset, temp_datapath)
    trainloader, valloader, trainvalloader, testloader = utils.get_dataset(cfg, temp_datapath, indexpaths)
    utils.set_device(cfg, verbose=False)
    if loadpath=='': # if loadpath is not specified, copy savepath
        loadpath=savepath
    # initialize counters
    current_itr = 0 # iteration for warmup steps tracking
    ckpt_epoch = 0 # initialize checkpoint epoch

    if run_train:
        # Define model and load checkpoint
        vitmodel = utils.get_model(cfg, report=True) # will report model summary and hyperparams if rank==0
        criterion, optimizer, weight_decayer, cosinescheduler, gradscaler = utils.get_optim_and_criterion(cfg, vitmodel, trainloader)
        vitmodel = DDP(vitmodel.to(cfg.RANK), device_ids=[rank], output_device=rank) # wrap model in Distributed Data Parallel
        # load checkpoint (if load_ckpt == "", don't load)
        ckpt_epoch, vitmodel, optimizer, weight_decayer, cosinescheduler, gradscaler = utils.load_checkpoint(cfg, load_ckpt, vitmodel, optimizer, weight_decayer, cosinescheduler, gradscaler)
        local_loss = utils.LocalWindow(maxsize=100) # local windowed loss calculator

        for epoch in range(ckpt_epoch, cfg.TRAIN.EPOCHS):
            trainloader.sampler.set_epoch(epoch)
            vitmodel.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = utils.unpack_data(data, cfg.TRAIN.DATASET, cfg.RANK, mixup)
                optimizer.zero_grad()
                weight_decayer.zero_grad()
                current_itr += 1
                if current_itr < cfg.TRAIN.WARMUP: # handle warmup LR
                    utils.adjust_lr(optimizer, cfg.TRAIN.LR * (current_itr+1)/cfg.TRAIN.WARMUP)
                    utils.copy_lr(optimizer, weight_decayer) # copy group['lr'] from src(left) to dst(right)
                with torch.cuda.amp.autocast(enabled=cfg.TRAIN.AMP, dtype=utils.get_dtype(cfg.MODEL.AMPDTYPE)):
                    if not cfg.TRAIN.DATASET.lower()[0:12] == "imagenet_dct": # if RGB model
                        outputs = vitmodel(inputs)
                    else: # if DCT model
                        outputs = vitmodel(inputs[0], inputs[1]) 
                    loss = criterion(outputs, labels)
                
                if cfg.TRAIN.AMP: # if using AMP
                    gradscaler.scale(loss).backward()
                    gradscaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(vitmodel.parameters(), max_norm=1)
                    gradscaler.step(optimizer)
                    gradscaler.step(weight_decayer)
                    gradscaler.update()
                    utils.clip_gradscaler(gradscaler)
                else: # if not using AMP
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(vitmodel.parameters(), max_norm=1)
                    optimizer.step()
                    weight_decayer.step()
                
                if current_itr >= cfg.TRAIN.WARMUP: # cosine scheduling
                    cosinescheduler.step() # decrease every iteration
                    utils.copy_lr(optimizer, weight_decayer) # copy group['lr'] from src(left) to dst(right)

                if cfg.RANK==0:
                    running_loss = local_loss.put(loss.item(), returnval=True)
                    current_lr = optimizer.param_groups[0]['lr']
                    train_num_batch = len(trainloader) 
                    if (i+1)%50==0:
                        writer.add_scalar(tag='Loss/Peritr_Train', scalar_value=running_loss, global_step = current_itr+1)
                    if verbose==2:
                        current_lr = optimizer.param_groups[0]['lr']
                        train_num_batch = len(trainloader) 
                        print(f'\r[Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Itr: {i+1}/{train_num_batch}] Loss: {running_loss:.4f} LR: {current_lr:.3e} ', end="", flush=True)
            val_acc, val_loss = eval.evaluate_model(cfg, vitmodel, valloader, criterion, verbose=False)
            tval_acc, tval_loss = eval.evaluate_model(cfg, vitmodel, trainvalloader, criterion, verbose=False)
            if cfg.RANK==0:
                if verbose==1 or verbose==2:
                    current_lr = optimizer.param_groups[0]['lr']
                    train_num_batch = len(trainloader) 
                    logging.info(f'\r[Epoch: {epoch+1}/{cfg.TRAIN.EPOCHS}, Itr: {i+1}/{train_num_batch}] Loss: {running_loss:.4f} LR: {current_lr:.3e}')
                    logging.info(f"\r[Val Acc: {val_acc*100:.2f} %, Val loss: {val_loss:.4f}, Train Acc (eval mode): {tval_acc*100:.2f} %, Train loss (eval mode): {tval_loss:.4f} ]")
                utils.write_tensorboard_train(writer, epoch, running_loss, val_loss, tval_loss, val_acc, tval_acc, optimizer.param_groups[0]['lr'])
                ckpt_savepath = utils.get_ckpt_path(savepath, epoch)
                utils.save_ckpt(ckpt_savepath, epoch, current_itr, vitmodel.module.state_dict(), optimizer.state_dict(), weight_decayer.state_dict(),
                                cosinescheduler.state_dict(), gradscaler.state_dict() if cfg.TRAIN.AMP else None, running_loss, val_acc, verbose>0)
            dist.barrier()

        if cfg.RANK==0:
            torch.save(vitmodel.module.state_dict(), savepath) # save vitmodel to savepath after training is done
            logging.info(f"Training complete. Saved model to {savepath}")

    if run_eval:
        if not run_train: # load model from `loadpath' if vitmodel is not trained
            vitmodel = utils.load_model_and_report(cfg, loadpath)
            criterion, optimizer, weight_decayer, cosinescheduler, gradscaler = utils.get_optim_and_criterion(cfg, vitmodel, trainloader)
        test_acc, test_loss = eval.evaluate_model(cfg, vitmodel, testloader, criterion, verbose=verbose>1)
        val_acc, val_loss, tval_acc, tval_loss = None, None, None, None
        if cfg.RANK==0 and (verbose==1 or verbose==2):
            logging.info(f"[Test Acc: {test_acc*100:.2f} %, Test loss: {test_loss:.4f} ]")

        if not run_train: # if not trained, evaluate on minival and train data as well
            val_acc, val_loss = eval.evaluate_model(cfg, vitmodel, valloader, criterion, verbose=verbose>1)
            tval_acc, tval_loss = eval.evaluate_model(cfg, vitmodel, trainvalloader, criterion, verbose=verbose>1)
            if cfg.RANK==0 and (verbose==1 or verbose==2):
                logging.info(f"[Val Acc: {val_acc*100:.2f} %, Val loss: {val_loss:.4f}, Train Acc (eval mode): {tval_acc*100:.2f} %, Train loss (eval mode): {tval_loss:.4f} ]")
        if cfg.RANK==0:
            utils.write_tensorboard_eval(writer, test_acc, test_loss, val_acc, val_loss, tval_acc, tval_loss)

    if benchmark > 0:
        if (not run_train) and (not run_eval): # if train or eval pipeline is not used, load model
            vitmodel = utils.load_model_and_report(cfg, loadpath)
            criterion, optimizer, weight_decayer, cosinescheduler, gradscaler = utils.get_optim_and_criterion(cfg, vitmodel, trainloader)
        # double check benchmark_model function
        eval.benchmark_model(cfg, vitmodel, benchmark, trainloader, testloader, mixup, verbose=verbose)

    if cfg.RANK==0:
        writer.close()
        if delete_dataset:
            shutil.rmtree(path_to_delete)
    dist.destroy_process_group()

def main():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    
    args = parse_args()

    cfg = configs.generate_config(
        modelarch = args.model_arch.lower(),
        domain = args.domain,
        modelver=args.embed_type,
        subblock=True if not args.no_subblock else False,
        epochs=None if args.epochs < 0 else args.epochs, # need to add
        batchsize=None if args.batch < 0 else args.batch, # need to change order
        lr=None if args.lr < 0 else args.lr,
        wd=None if args.wd < 0 else args.wd,
        drop=None if args.drop < 0 else args.drop,
        warmup_steps=None if args.warmup_steps < 0 else args.warmup_steps, # need to add
        auglist=None if args.ops_list == '' else args.ops_list.split(","),
        num_ops=None if args.num_ops < 0 else args.num_ops, # need to add
        ops_magnitude=None if args.ops_magnitude < 0 else args.ops_magnitude, # need to add
        seed=None if args.seed < 0 else args.seed, # need to add
        amp=None if args.amp < 0 else args.amp,
        ampdtype=None if args.ampdtype == '' else args.ampdtype,
        use_msrsync=args.use_msrsync,
    )

    utils.copy_dataset(args.datapath, args.temp_datapath, args.indexpaths.split(","),
                       args.num_cpus, args.no_extract, args.no_resize, args.use_msrsync, args.verbose>1)

    max_avail_gpus = torch.cuda.device_count()
    if args.num_gpus < 0:
        world_size = max_avail_gpus # number of GPUs
    else:
        assert args.num_gpus <= max_avail_gpus, f"Number of GPUs cannot exceed maximum available GPUs on this machine (max: {max_avail_gpus}, requested: {args.num_gpus})"
        world_size = args.num_gpus # use specified number of GPUs
    logging.info(f"Spawning processes on {world_size} GPUs.")

    torchmp.spawn(
        traineval,
        args=( # rank is appended at the start
            world_size,
            cfg,

            args.temp_datapath,
            args.indexpaths.split(","),
            args.delete_dataset,
            args.num_cpus,

            args.train,
            args.eval,
            args.benchmark,
            args.savepath,
            args.loadpath,
            args.load_ckpt,
            args.deterministic,
            args.verbose,
            args.port,
        ),
        nprocs=world_size,
    )

if __name__ == "__main__":
    main()