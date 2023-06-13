# Benchmark models

import torch
from torch import nn
import torch.optim as optim
from time import time
import logging
from fvcore.nn import FlopCountAnalysis ## thank you Meta

def benchmark_dataloader(dataloader, itr:int =100, device='cpu', dtype=torch.float32, mixup=None, verbose=True):
    """
    Run benchmark for `itr' iterations on dataloader
    Args:
        dataloader: pytorch dataloader object
        itr: iteration limit
        device: device (cpu, cuda, or cuda index)
        dtype: dtype (default: torch.float32)
        verbose: Print result
    
    Returns:
        time_took: time took in seconds
        img_count: number of processed images
    """
    inittime=time()
    idx = 0 # tracks current index
    img_count = 0
    init_took = 0
    time_took = 0
    while idx < itr:
        for i, data in enumerate(dataloader):
            if i == 0:
                starttime = time()
            if idx >= itr:
                break
            inputs, labels = data
            labels = labels.to(device)
            img_count += data[1].shape[0]
            if type(inputs) == list or type(inputs) == set:
                # DCT
                y = inputs[0]
                cbcr = inputs[1]
                y = y.to(device)
                cbcr = cbcr.to(device)
            else:
                # RGB
                inputs = inputs.to(device) # send to gpu
            if verbose:
                print(f"\rBenchmarking dataloader... {idx+1}/{itr}", end="", flush=True)
            idx += 1
        endtime = time()
        time_took += endtime - starttime
        init_took += starttime - inittime
        inittime = time()
    if verbose:
        print("\n", end="")
        logging.info(f"   Dataloader took: {time_took:.2f} sec for {idx} itrs / {img_count} imgs. {img_count/time_took:.2f} FPS. Init took {init_took:.2f} sec")
    return time_took, img_count

def benchmark_modelfbp_rgb(model, Imgshape=(128, 3, 224, 224), outshape=(128, ),
                        itr:int =100, mode='fbp', criterion=nn.CrossEntropyLoss(), use_amp=False, mixup=None,
                        device='cpu', dtype=torch.float32, verbose=True):
    """
    Run benchmark for `itr' iterations on model forward/backward pass (dct)
    Args:
        model: pytorch model
        Imgshape (tuple): model input shape (including batch dimension)
        itr: iteration limit
        criterion: dummy criterion to use (default: nn.CrossEntropyLoss)
        use_amp: use automatic mixed precision
        device: device (cpu, cuda, or cuda index)
        dtype: dtype (default: torch.float32)
        verbose: print result if true
    
    Return:
        time_took: time took in seconds
        img_count: number of processed images
    """
    dummy_data = torch.randn(Imgshape, device="cpu", dtype=dtype)
    dummy_out = torch.randint(0, 999, outshape, device="cpu", dtype=torch.int64)
    dummy_data = dummy_data.to(device)
    dummy_out = dummy_out.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0, eps=1e-4) 
    img_count = 0
    if mode=="fwd":
        dummy_data_clone = dummy_data.clone()
        dummy_out_clone = dummy_out.clone()
    inittime=time()
    if use_amp:
        gradscaler = torch.cuda.amp.GradScaler()
    with torch.set_grad_enabled(mode=='fbp'):
        for i in range(itr):
            if i == 0:
                starttime = time()
            if mode=='fbp':
                optimizer.zero_grad()
                dummy_data_clone = dummy_data.clone()
                dummy_out_clone = dummy_out.clone()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                if mode=="fbp" and mixup:
                    dummy_data_clone, dummy_out_clone = mixup(dummy_data_clone, dummy_out_clone)
                output = model(dummy_data_clone)
                loss = criterion(output, dummy_out_clone)
                if mode=='fbp':
                    if use_amp:
                        gradscaler.scale(loss).backward()
                        gradscaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        gradscaler.step(optimizer)
                        gradscaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        optimizer.step()
            img_count += outshape[0]
            if verbose:
                print(f"\rBenchmarking model F/B pass... {i+1}/{itr}", end="", flush=True)
    endtime=time()
    init_took = starttime-inittime
    time_took = endtime-starttime
    if verbose:
        print("\n", end="")
        logging.info(f"   Model F/B pass (mode:{mode}, amp:{use_amp}) took: took: {time_took:.2f} sec for {itr} itrs / {img_count} imgs. {img_count / time_took:.2f} FPS. Init took {init_took:.2f} sec")
    return time_took, img_count

def benchmark_modelfbp_dct(model, Yshape=(128, 28, 28, 1, 8, 8), Cshape=(128, 14, 14, 2, 8, 8), outshape=(128, ),
                        itr:int =100, mode='fbp', criterion=nn.CrossEntropyLoss(), use_amp=False, mixup=None,
                        device='cpu', dtype=torch.float32, verbose=True):
    """
    Run benchmark for `itr' iterations on model forward/backward pass (dct)
    Args:
        model: pytorch model
        Yshape (tuple): model input shape (Y)(including batch dimension)
        Cshape (tuple): model input shape (CbCr)
        itr: iteration limit
        mode: 'fbp' or 'fwd'
        criterion: dummy criterion to use (default: nn.CrossEntropyLoss)
        use_amp: use amp if true
        device: device (cpu, cuda, or cuda index)
        dtype: dtype (default: torch.float32)
        verbose: print result if true
    
    Return:
        time_took: time took in seconds
        img_count: number of processed images
    """
    dummy_data_Y = torch.randn(Yshape, device="cpu", dtype=dtype)
    dummy_data_C = torch.randn(Cshape, device="cpu", dtype=dtype)
    dummy_out = torch.randint(0, 999, outshape, device="cpu", dtype=torch.int64)
    dummy_data_Y = dummy_data_Y.to(device)
    dummy_data_C = dummy_data_C.to(device)
    dummy_out = dummy_out.to(device)
    assert dummy_out.ndim == 1, f"Dummy out should have one dimension. Current: {dummy_out.shape}, {dummy_out.ndim}"
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0, eps=1e-4)
    img_count=0 
    if mode=="fwd":
        dummy_data_Y_clone = dummy_data_Y.clone()
        dummy_data_C_clone = dummy_data_C.clone()
        dummy_out_clone = dummy_out.clone()
    inittime=time()
    if use_amp:
        gradscaler = torch.cuda.amp.GradScaler()
    with torch.set_grad_enabled(mode=='fbp'):
        for i in range(itr):
            if i == 0:
                starttime = time()
            if mode=='fbp':
                optimizer.zero_grad()
                dummy_data_Y_clone = dummy_data_Y.clone()
                dummy_data_C_clone = dummy_data_C.clone()
                dummy_out_clone = dummy_out.clone()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                if mode=="fbp" and mixup:
                    (dummy_data_Y_clone, dummy_data_C_clone), dummy_out_clone = \
                        mixup((dummy_data_Y_clone, dummy_data_C_clone), dummy_out_clone)
                output = model(dummy_data_Y_clone, dummy_data_C_clone)
                loss = criterion(output, dummy_out_clone)
                if mode=='fbp':
                    if use_amp:
                        gradscaler.scale(loss).backward()
                        gradscaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        gradscaler.step(optimizer)
                        gradscaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        optimizer.step()
            img_count += outshape[0]
            if verbose:
                print(f"\rBenchmarking model F/B pass... {i+1}/{itr}", end="", flush=True)
    endtime=time()
    init_took = starttime-inittime
    time_took = endtime-starttime
    if verbose:
        print("\n", end="")
        logging.info(f"   Model F/B pass (mode:{mode}, amp:{use_amp}) took: {time_took:.2f} sec for {itr} itrs / {img_count} imgs. {img_count / time_took:.2f} FPS. Init took {init_took:.2f} sec")
    return time_took, img_count

def benchmark_mixup_rgb(Imgshape=(128, 3, 224, 224), outshape=(128, ),
                        itr:int =100, mixup=None,
                        device='cpu', dtype=torch.float32, verbose=True):
    """
    Run benchmark for `itr' iterations on model forward/backward pass (dct)
    Args:
        model: pytorch model
        Imgshape (tuple): model input shape (including batch dimension)
        itr: iteration limit
        criterion: dummy criterion to use (default: nn.CrossEntropyLoss)
        use_amp: use automatic mixed precision
        device: device (cpu, cuda, or cuda index)
        dtype: dtype (default: torch.float32)
        verbose: print result if true
    
    Return:
        time_took: time took in seconds
        img_count: number of processed images
    """
    dummy_data = torch.randn(Imgshape, device="cpu", dtype=dtype)
    dummy_out = torch.randint(0, 999, outshape, device="cpu", dtype=torch.int64)
    img_count = 0
    time_took = 0
    init_took = 0
    inittime=time()
    starttime = time()
    if mixup==None:
        return 1,1
    for i in range(itr):
        dummy_data_clone = dummy_data.clone()
        dummy_out_clone = dummy_out.clone()
        dummy_data_clone = dummy_data_clone.to(device)
        dummy_out_clone = dummy_out_clone.to(device)
        img_count += outshape[0]
        if verbose:
            print(f"\rBenchmarking memory copy ... {i+1}/{itr}", end="", flush=True)
    endtime=time()
    time_took += endtime-starttime
    init_took = starttime-inittime
    if verbose:
        print("\n", end="")
        logging.info(f"   Mem Transfer (RGB) took: took: {time_took:.3f} sec for {itr*10} itrs / {img_count} imgs. {time_took/img_count*1000:.3f} ms/img. Init took {init_took:.2f} sec")
    return time_took, img_count

def benchmark_mixup_dct(Yshape=(128, 28, 28, 1, 8, 8), Cshape=(128, 14, 14, 2, 8, 8), outshape=(128, ),
                        itr:int =100, mixup=None,
                        device='cpu', dtype=torch.float32, verbose=True):
    """
    Run benchmark for `itr' iterations on model forward/backward pass (dct)
    Args:
        model: pytorch model
        Imgshape (tuple): model input shape (including batch dimension)
        itr: iteration limit
        criterion: dummy criterion to use (default: nn.CrossEntropyLoss)
        use_amp: use automatic mixed precision
        device: device (cpu, cuda, or cuda index)
        dtype: dtype (default: torch.float32)
        verbose: print result if true
    
    Return:
        time_took: time took in seconds
        img_count: number of processed images
    """
    dummy_data_Y = torch.randn(Yshape, device="cpu", dtype=dtype)
    dummy_data_C = torch.randn(Cshape, device="cpu", dtype=dtype)
    dummy_out = torch.randint(0, 999, outshape, device="cpu", dtype=torch.int64)
    img_count = 0
    time_took=0
    init_took=0
    inittime=time()
    starttime = time()
    if mixup == None:
        return 1, 1
    for i in range(itr):
        dummy_data_Y_clone = dummy_data_Y.clone()
        dummy_data_C_clone = dummy_data_C.clone()
        dummy_out_clone = dummy_out.clone()
        dummy_data_Y_clone = dummy_data_Y_clone.to(device)
        dummy_data_C_clone = dummy_data_C_clone.to(device)
        dummy_out_clone = dummy_out_clone.to(device)
        img_count += outshape[0]
        if verbose:
            print(f"\rBenchmarking mixup ... {i+1}/{itr}", end="", flush=True)
    endtime=time()
    time_took = endtime-starttime
    init_took = starttime-inittime
    if verbose:
        print("\n", end="")
        logging.info(f"   Mem Transfer (DCT) took: took: {time_took:.3f} sec for {itr*10} itrs / {img_count} imgs. {time_took/img_count*1000:.3f} ms/img. Init took {init_took:.2f} sec")
    return time_took, img_count

def benchmark_pipeline(model, dataloader, itr, criterion=nn.CrossEntropyLoss(), use_amp=False, mixup=None,
    mode='train', modeltype='dct', device='cpu', dtype=torch.float32, verbose=True):
    """
    Run benchmark of a trainig pipeline .
    Args:
        model: pytorch model
        dataloader: dataloader
        itr: iteration limit
        criterion: dummy criterion to use (default: nn.CrossEntropyLoss)
        mode: 'train' or 'test' -- 'test' doesn't do backprop
        modeltype: Model type (DCT or RGB)
        device: device (cpu, cuda, or cuda index)
        dtype: dtype (default: torch.float32)
        verbose: print result if true

    Return:
        time_took: time took in seconds
    """
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0, eps=1e-4) 
    inittime=time()
    idx = 0 # tracks current index
    init_took=0
    time_took=0
    img_count=0
    if use_amp:
        gradscaler = torch.cuda.amp.GradScaler()
    with torch.set_grad_enabled(mode=='train'):
        while idx < itr:
            for i, data in enumerate(dataloader):
                if i == 0:
                    starttime = time()
                if idx >= itr:
                    break
                inputs, labels = data
                labels = labels.to(device)
                if mode=='train':
                    optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    if modeltype=='dct':
                        y = inputs[0]
                        cbcr = inputs[1]
                        y = y.to(device)
                        cbcr = cbcr.to(device)
                        if mode=="train" and mixup:
                            (y, cbcr), labels = mixup((y, cbcr), labels)
                        output = model(y, cbcr)
                    else:
                        inputs = inputs.to(device)
                        if mode=="train" and mixup:
                            inputs, labels = mixup(inputs, labels)
                        output = model(inputs)
                    loss = criterion(output, labels)
                    if mode=='train':
                        if use_amp:
                            gradscaler.scale(loss).backward()
                            gradscaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                            gradscaler.step(optimizer)
                            gradscaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                            optimizer.step()
                    img_count += labels.shape[0]
                    if verbose:
                        print(f"\rBenchmarking pipeline... {idx+1}/{itr}", end="", flush=True)
                    idx += 1
            endtime = time()
            time_took += endtime - starttime
            init_took += starttime - inittime            
            inittime = time()
    if verbose:
        print("\n", end="")
        logging.info(f"   Pipeline (amp: {use_amp}) took: {time_took:.2f} sec for {idx} itrs / {img_count} imgs. {img_count / time_took:.2f} FPS. Init took {init_took:.2f} sec")
    return time_took, img_count