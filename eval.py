import torch
import torch.distributed as dist
import torchmetrics
import logging
import utils.pipeline_utils as utils
import benchmark as bm

def evaluate_model(cfg, model, dataloader, criterion, verbose=False):
    """
    Evaluate model

    Inputs:
        cfg: yacs config node
        model: PyTorch model to evaluate
        dataloader: PyTorch dataloader
        crietrion: Criterion to evaluate on (ex. crossentropyloss)
        verbose: If true, report progress for rank 0 process (device_rank == 0)
    
    Outputs:
        Accuracy: Accuracy on dataset (float)
        Loss: Loss on dataset (float)
    """
    # Read parameters from config
    dataset_name = cfg.TRAIN.DATASET
    device_rank = cfg.RANK
    use_amp = cfg.TRAIN.AMP
    num_classes = cfg.MODEL.CLASSES

    num_batch = len(dataloader)
    total_loss = torch.zeros(1, dtype=torch.float32, device=device_rank)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1, dist_sync_on_step=True, process_group=None).to(device_rank)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, labels = utils.unpack_data(data, dataset_name, device_rank) # unpack data accordingly for RGB or DCT
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16): # use amp during evaluation if specified
                if dataset_name.lower()[0:12] == 'imagenet_dct':
                    outputs = model(images[0], images[1])
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            acc.forward(outputs, labels)
            if verbose and device_rank==0:
                print(f'\rEvaluating... {i+1}/{num_batch}   ', end="", flush=True)
    # average loss across GPUs
    dist.reduce(total_loss, dst=0, op=dist.ReduceOp.AVG)
    epoch_loss = total_loss.item()
    epoch_loss /= (i+1) # average loss across all data

    return acc.compute().item(), epoch_loss

def benchmark_model(
    cfg, model, iterations, trainloader, testloader, mixup=None, nomixup=False, verbose=0
    ):
    """
    Benchmarks model
        1. Train/Test dataloader throughput
        2. Model forward/backward throughput
        3. Pipeline (dataloader + model fwd/bwd) throughput

    Args
    Inputs:
        cfg: yacs config
        model: Pytorch model to benchmark
        iterations: number of iterations to benchmark
        trainloader: train dataloader
        testloader: test dataloader
        mixup: mixup for train dataloader (default: None)
        nomixup: If true, don't apply mixup during benchmark (on train dataloader)
        verbose (int): 0, 1: No intermediate prints, 2: print every iterations

    Outputs: (averaged per GPU)
        Train dataloader throughput (FPS/GPU)
        Test dataloader throughput
        Model forward/backword throughput
        Model forward throughput
        Train pipeline throughput
        Test pipeline throughput
    """
    # Read parameters from config
    batch_size = cfg.TRAIN.BATCHPERGPU
    dataset_name = cfg.TRAIN.DATASET
    device_rank = cfg.RANK
    use_amp = cfg.TRAIN.AMP

    # initialize tensors to store benchmark results. Will use torch.dist to average them across gpus
    trainloader_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    testloader_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    modelfbp_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    modelfwd_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    trainpl_time = torch.zeros(1, dtype=torch.float32, device=device_rank)
    testpl_time = torch.zeros(1, dtype=torch.float32, device=device_rank)

    imgs_tr = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_test = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_fbp = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_fwd = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_trp = torch.zeros(1, dtype=torch.int64, device=device_rank)
    imgs_tsp = torch.zeros(1, dtype=torch.int64, device=device_rank)

    # Benchmark dataloader
    trainloader_time[0], imgs_tr[0] = bm.benchmark_dataloader(trainloader, iterations, 
        device=device_rank, dtype=torch.float32, mixup=None if nomixup else mixup, verbose=(verbose>1) and device_rank==0)
    
    testloader_time[0], imgs_test[0] = bm.benchmark_dataloader(testloader, iterations, 
        device=device_rank, dtype=torch.float32, verbose=(verbose>1) and device_rank==0)

    if (verbose>0) and device_rank==0:
        logging.info("Dataloader benchmark done")
    
    # Benchmark model forward / backward
    if dataset_name.lower()[0:12]=='imagenet_dct':
        modelfbp_time[0], imgs_fbp[0] = bm.benchmark_modelfbp_dct(
            model, (batch_size, 1, 28, 28, 8, 8), (batch_size, 2, 14, 14, 8, 8), (batch_size, ), 
            max(iterations, 300), 'fbp', use_amp=use_amp, mixup=None if nomixup else mixup,
            device=device_rank, dtype=torch.float32, verbose=(verbose>1) and device_rank==0)
        modelfwd_time[0], imgs_fwd[0] = bm.benchmark_modelfbp_dct(
            model, (batch_size, 1, 28, 28, 8, 8), (batch_size, 2, 14, 14, 8, 8), (batch_size, ), 
            max(iterations, 300), 'fwd',use_amp=use_amp, mixup=None, 
            device=device_rank, dtype=torch.float32, verbose=(verbose>1) and device_rank==0)
        modeltype = 'dct'
    else:
        modelfbp_time[0], imgs_fbp[0] = bm.benchmark_modelfbp_rgb(
            model, (batch_size, 3, 224, 224), (batch_size, ), max(iterations, 300), 'fbp', use_amp=use_amp, 
            mixup=None if nomixup else mixup, device=device_rank, dtype=torch.float32, verbose=(verbose>1) and device_rank==0)
        model[0], imgs_fwd[0] = bm.benchmark_modelfbp_rgb(
            model, (batch_size, 3, 224, 224), (batch_size, ), max(iterations, 300), 'fwd', use_amp=use_amp, 
            mixup=None, device=device_rank, dtype=torch.float32, verbose=(verbose>1) and device_rank==0)
        modeltype = 'rgb'

    if (verbose>0) and device_rank==0:
        logging.info("Model forward/backward benchmark done")

    # Benchmark pipeline
    trainpl_time[0], imgs_trp[0] = bm.benchmark_pipeline(
        model, trainloader, iterations, use_amp=use_amp, mixup=None if nomixup else mixup, 
        mode='train', modeltype=modeltype, device=device_rank, dtype=torch.float32, verbose=(verbose>1) and device_rank==0
    )
    testpl_time[0], imgs_tsp[0] = bm.benchmark_pipeline(
        model, testloader, iterations, use_amp=use_amp, mixup=None,
        mode='test', modeltype=modeltype, device=device_rank, dtype=torch.float32, verbose=(verbose>1) and device_rank==0
    )

    if (verbose>0) and device_rank==0:
        logging.info("Model forward/backward benchmark done")
    dist.barrier() # wait for all processes to complete benchmark

    # Reduce benchmark results across all gpu processes
    dist.reduce(trainloader_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(testloader_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_tr, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_test, 0, op=dist.ReduceOp.SUM)

    dist.reduce(modelfbp_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(modelfwd_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_fbp, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_fwd, 0, op=dist.ReduceOp.SUM)

    dist.reduce(trainpl_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(testpl_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_trp, 0, op=dist.ReduceOp.SUM)
    dist.reduce(imgs_tsp, 0, op=dist.ReduceOp.SUM)

    trainloader_fps = imgs_tr[0]/trainloader_time[0]
    testloader_fps = imgs_test[0]/testloader_time[0]
    modelfbp_fps = imgs_fbp[0]/modelfbp_time[0]
    modelfwd_fps = imgs_fwd[0]/modelfwd_time[0]
    trainpipe_fps = imgs_trp[0]/trainpl_time[0]
    testpipe_fps = imgs_tsp[0]/testpl_time[0]

    if (verbose>0) and device_rank==0:
        logging.info(f"Train loader: {trainloader_fps:.2f} FPS per GPU")
        logging.info(f"Model F/B pass: {modelfbp_fps:.2f} FPS per GPU")
        logging.info(f"Train pipeline: {trainpipe_fps:.2f} FPS per GPU")

        logging.info(f"Test loader: {testloader_fps:.2f} FPS per GPU")
        logging.info(f"Model Fwd pass: {modelfwd_fps:.2f} FPS per GPU")
        logging.info(f"Test pipeline: {testpipe_fps:.2f} FPS per GPU")
    
    return trainloader_fps, testloader_fps, modelfbp_fps, modelfwd_fps, trainpipe_fps, testpipe_fps