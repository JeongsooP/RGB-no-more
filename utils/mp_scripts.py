import PIL.Image
import os
import multiprocessing as mp
import pandas as pd
import numpy as np
import tarfile
from time import time, sleep
import logging

class ImageResizer():
    ## Resizes images into a specified size
    def __init__(self, size=256, num_proc=4, indexpath="indexfile.csv", basepath="", output_basepath="", num_img=-1, verbose=False):
        self.num_proc = num_proc
        self.size = size
        self.lock = mp.Lock()
        self.queue = mp.Queue()
        self.processes=[]
        self.verbose = verbose
        self.imageindex = pd.read_csv(indexpath).to_numpy()
        indexlength = num_img if num_img > 0 else len(self.imageindex)
        self.idxsplit = np.linspace(0, indexlength, num_proc+1, dtype=int) # simple implementation
        for i in range(len(self.imageindex)):
            self.queue.put(self.imageindex[i, 0])
        self.basepath=basepath
        self.output_basepath=output_basepath
    
    @staticmethod
    def resize_image(id, size, lock, imageindex, start, end, basepath, output_basepath, verbose):
        """
        Resize image to specified size / index-based
        """
        if verbose:
            start_time = time()
            logging.info(f"Process {id} started resizing image index from: {start} to {end}.")
            idxrange = end-start
            printed = False
        for i in range(start, end):
            img = PIL.Image.open(os.path.join(basepath, imageindex[i, 0])).convert('RGB')
            img = img.resize((size, size), resample=PIL.Image.Resampling.BILINEAR)
            output_path = os.path.join(output_basepath, imageindex[i, 0])
            dirname = os.path.dirname(output_path)
            if not os.path.isdir(dirname):
                with lock:
                    if not os.path.isdir(dirname):
                        os.makedirs(dirname)
            img.save(output_path)
            if verbose and (idxrange//2 < i-start) and not printed:
                logging.info(f"Process {id} is halfway done: {i-start}/{idxrange}")
                printed = True
        if verbose:
            end_time = time() 
            logging.info(f"Process {id} ended resizing image index from: {start} to {end}. This took {end_time-start_time:.0f}s")
    
    @staticmethod
    def resize_image_queue(id, size, lock, queue, basepath, output_basepath, verbose):
        """
        Resize image to specified size / queue-based
        """
        if verbose:
            rank = mp.current_process()._identity[0]
            logging.info(f"Process {id} started resizing image on core # {rank}")
        while True:
            lock.acquire()
            if queue.empty():
                if verbose:
                    logging.info(f"Process {id}: queue is empty. Exiting...")
                lock.release()
                break
            else:
                imgname = queue.get()
                lock.release()

            img = PIL.Image.open(os.path.join(basepath, imgname)).convert('RGB')
            img = img.resize((size, size), resample=PIL.Image.Resampling.BILINEAR)
            output_path = os.path.join(output_basepath, imgname)
            dirname = os.path.dirname(output_path)
            if not os.path.isdir(dirname):
                with lock:
                    if not os.path.isdir(dirname):
                        os.makedirs(dirname)
            img.save(output_path)
    
    def start_processes(self):
        if self.verbose:
            logging.info(f'Starting resize with {self.num_proc} processes...')
        for i in range(self.num_proc):
            self.processes.append(
                mp.Process(
                    #target=self.resize_image, args=(
                    #    i, self.size, self.lock, self.imageindex, self.idxsplit[i], self.idxsplit[i+1], self.basepath, self.output_basepath, self.verbose
                    #    )
                    target=self.resize_image_queue, args=(i, self.size, self.lock, self.queue, self.basepath, self.output_basepath, self.verbose)
                    )
                )
            self.processes[i].start()
    
    def end_processes(self, waittime:int=10):
        for i in range(waittime, 0, -1):
            logging.info(f"\rTerminating processes in {i}...", end="", flush=True)
            sleep(1)
        for proc in self.processes:
            proc.terminate()
        logging.info("Processes terminated")
    
    def wait_and_terminate(self):
        starttime=time()
        for proc in self.processes:
            proc.join()
            proc.close()
        endtime=time()
        time_took = endtime-starttime
        if self.verbose:
            logging.info(f"All processes terminated (img resizing). It took {time_took/60:.1f} mins")

class tar_extractor():
    """
    Extracts tar files of ImageNet train data using multiprocessing

    Args:
        num_proc: number of total available CPUs
        files: list of files to use multiprocessing on
        imgnet_trainpath: path to imagenet train which contains the .tar files
    """
    def __init__(self, num_proc=4, files=[], imgnet_trainpath='/<TMP>/imgnet_data/train', verbose=False):
        self.num_proc = num_proc
        self.processes=[]
        self.files=files
        self.idxsplit = np.linspace(0, len(files), num_proc+1, dtype=int)
        self.imgnet_trainpath = imgnet_trainpath
        self.verbose = verbose

        self.queue = mp.Queue() # multiprocessing queue
        self.lock = mp.Lock() # multiprocessing lock
        for tarname in files:
            self.queue.put(tarname)

    @staticmethod
    def extract_tar(id, files, imgnet_trainpath, start, end, verbose):
        """
        Extracts tar files in `files' from `start' to `end' index.

        Args:
            id: multiprocessing id
            files: list of tar files
            imgnet_trainpath: path to imgnet train which contains the .tar files
            start: start index of file to extract
            end: end index of file to extract
            verbose: If true, report progress
        """
        if verbose:
            filelen=len(files)
            logging.info(f"Process {id} started extracting tar files (total: {filelen}) from idx {start} to {end}.")
        for i in range(start, end):
            tarname = files[i]
            #if tarname[-4:].lower()=='.tar': # already checked by the parent processor so it is redundant. flagged to be deleted.
            tarname_folder = os.path.join(imgnet_trainpath, os.path.basename(tarname).split('.')[0])
            if not os.path.isdir(tarname_folder):
                os.makedirs(tarname_folder)
            child_tar = tarfile.open(os.path.join(imgnet_trainpath, tarname))
            child_tar.extractall(tarname_folder)
            os.remove(os.path.join(imgnet_trainpath, tarname))

    @staticmethod
    def extract_tar_queue(id, imgnet_trainpath, queue, lock, verbose):
        """
        Gets `tarname' from `queue'.
        Extracts `tarname' tar file in `imgnet_trainpath' and extract it to the same folder.
        If `queue' is empty, return.

        Args:
            id: multiprocessing id
            tarname: tar file name
            imgnet_trainpath: path to imgnet train which contains the .tar files
            queue: Python multiprocessing queue
            lock: Python multiprocessing lock
        """
        if verbose:
            logging.info(f"Process {id} started.")
        while True:
            lock.acquire()
            if queue.empty():
                if verbose:
                    logging.info(f"Process {id}: queue is empty. Exiting...")
                lock.release()
                break
            else:
                tarname = queue.get()
                lock.release()

            tarname_folder = os.path.join(imgnet_trainpath, os.path.basename(tarname).split('.')[0])
            if not os.path.isdir(tarname_folder):
                os.makedirs(tarname_folder)
            child_tar = tarfile.open(os.path.join(imgnet_trainpath, tarname))
            child_tar.extractall(tarname_folder)
            os.remove(os.path.join(imgnet_trainpath, tarname))

    def start_processes(self):
        for i in range(self.num_proc):
            self.processes.append(
                mp.Process(
                    #target=self.extract_tar, args=(i, self.files, self.imgnet_trainpath, self.idxsplit[i], self.idxsplit[i+1], self.verbose)
                    #)
                    target=self.extract_tar_queue, args=(i, self.imgnet_trainpath, self.queue, self.lock, self.verbose)
                    )
                )
            self.processes[i].start()
    
    def end_processes(self, waittime:int=10):
        for i in range(waittime, 0, -1):
            logging.info(f"\rTerminating processes in {i}...", end="", flush=True)
            sleep(1)
        for proc in self.processes:
            proc.terminate()
        if self.verbose:
            logging.info("Processes terminated")

    def wait_and_terminate(self):
        starttime=time()
        for proc in self.processes:
            proc.join()
        endtime=time()
        time_took = endtime-starttime
        if self.verbose:
            logging.info(f"All processes terminated (tar extractor). It took {time_took/60:.1f} mins")

def main():
    # example
    imageresizer = ImageResizer(
        size=256, num_proc=20, indexpath="indexfile.csv", basepath="", output_basepath="", num_img=10, verbose=True
    )

if __name__ == "__main__":
    main()