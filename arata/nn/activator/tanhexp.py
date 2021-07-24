import warnings
import torch
from torch import Tensor
from torch.nn import Module

@torch.jit.script
def tanhexp(input: Tensor) -> Tensor:

    """out = x * tanh(exp(x)) """
    return torch.mul(input, torch.tanh(torch.exp(input)))