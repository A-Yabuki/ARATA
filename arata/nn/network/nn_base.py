# coding: utf-8

from abc import ABC, abstractmethod

import torch

class NNBase(ABC):
    """Neural network architecture base class"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()