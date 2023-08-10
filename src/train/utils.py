import torch
import gc
import multiprocessing


def clear_gpu():
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()


def get_num_processors():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    num_processors = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1
    return num_processors
 