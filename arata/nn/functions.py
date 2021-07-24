# coding: utf-8

import torch
from torch.autograd import Variable

def try_gpu(e: Variable):
    if torch.cuda.is_available():
        # non_blocking...
        # whether cpu is allow to other tasks, 
        # while data transmission between CPU and GPU. 
        return e.to('cuda', non_blocking=True)
        
    return e