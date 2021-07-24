# coding: utf-8

import os
from abc import ABC
from typing import Callable

import torch
import torch.cuda

from arata.common.enum import ErrorType
from arata.common.error_handler import ChainedErrorHandler
from . import functions
from .network import deeplab_v3plus_xception, initial_weight

class NNClientBase(ABC):

    r"""
    Neural Network Base Class
    offeres the common network construction process 
    for subclasses created for more special objective.
    """

    def __init__(self) -> None:
        self.net = None


    def construct_network(
        self, 
        num_class: int, 
        num_middle_layer: int, 
        middle_layer_scale: int, 
        normalizer: Callable, 
        activator: Callable) -> None:

        r"""
        Constructs the network architecture.

        Args:
            num_class (int): The number of classes.
            num_middle_layer (int): The number of the middle layer of deep lab v3+
            middle_layer_scale (int): The scale of the middle layer of deep lab v3+
            normalizer (Callable): normalization function of the Deep lab v3+
            activator (Callable): activation function of the Deep lab v3+
        """
        
        deeplab_v3plus_xception.DeepLabSettings.num_class = int(num_class)
        deeplab_v3plus_xception.DeepLabSettings.num_middle_layer = int(num_middle_layer)
        deeplab_v3plus_xception.DeepLabSettings.middle_layer_scale = int(middle_layer_scale)
        deeplab_v3plus_xception.DeepLabSettings.normalizer = normalizer
        deeplab_v3plus_xception.DeepLabSettings.activator = activator

        deeplab = deeplab_v3plus_xception.DeepLabv3plusXception()
        self.net = functions.try_gpu(deeplab)


    def initialize_network(self, params_file_path: str) -> None:
        
        r"""
        Initialize network parameters.

        Args:
            params_file_path (str): path of a snapshot file or model file.
        """
        
        if isinstance(params_file_path, str) and (len(params_file_path)!=0):
            try:
                if os.path.exists(params_file_path):
                    self.net.load_state_dict(torch.load(params_file_path), strict=False)
                    return

            except:
                ChainedErrorHandler(
                    "Failed to load the model. Please check network architecture settings.", 
                    ErrorType.INVALID_INPUT_VALUE_ERROR)

        init_w = initial_weight.Initial_weight(nonlinearity='relu')
        self.net.apply(init_w.w_henorm)