# coding: utf-8

import cv2
import numpy as np
from typing import Any, List

import torch
import torch.cuda
import torch.nn.functional as F

from arata.common.enum import ErrorType
from arata.common.wrapper_collection import watch_error
from . import functions
from .net import NNClientBase
from .network.nn_base import NNBase

class Predictor(NNClientBase):

    r"""
    Detects roots in an input image.
    """

    def __init__(self) -> None:
        super().__init__()


    def predict(self, inputs: 'np.ndarray[np.uint8] (H, W, 3)', 
                multi_scale: bool = False) -> List['np.ndarray[np.uint8] (H, W, 3)']:

        r"""
        Detects roots in an input image.

        Args:
            inputs (List(Variable)): inputs
            multi_scale (bool): flag indicating using multi scale input or not.
        """

        self.net.eval()
        scales = [0.75, 1, 1.25] if multi_scale else [1]
        outputs = []

        with torch.no_grad():
            for i, input in enumerate(inputs):                
                pred = self._predict_multi_scale(input, self.net, scales)
                outputs.append(pred)

        return outputs


    @watch_error(
        "Failed to trace roots.",
        ErrorType.CRITICAL_ERROR
        )
    def _predict_multi_scale(self, input: 'np.ndarray[np.uint8] (H, W, 3)', 
                             model: NNBase, scales: List[float]) -> 'np.ndarray[np.uint8] (H, W, 3)':

        outputs = []
        
        for scale in scales:

            resized_input = self._multi_scaler(input, scale)
            pred = self._predict(resized_input, model)
            
            original_size_pred = self._multi_scaler(pred, 1/scale)
            original_h, original_w = input.shape[:2]
            pred_h, pred_w = original_size_pred.shape[:2]

            # multiscale時、元のサイズに戻しても端数が発生する事があるので、
            # エラーにならないよう整形
            if (original_h != pred_h or original_w != pred_w):
                original_size_pred[:original_h, :original_w, :]
            
            outputs.append(original_size_pred)

        return np.mean(outputs, axis=0).astype(np.uint8)


    @watch_error(
        "Failed to trace roots.",
        ErrorType.CRITICAL_ERROR
        )
    def _predict(self, input: 'np.ndarray[np.uint8] (H, W, 3)', model: NNBase) -> 'np.ndarray[np.uint8] (H, W, 3)':
    
        r"""
        input:  size...(h, w, 3)
        output: size...(h, w, classNum) 
        """
        input = input.astype(np.float)
        input /= 255
                
        input = functions.try_gpu(torch.tensor(input.transpose(2, 0, 1)[np.newaxis,:,:,:], dtype=torch.float))

        output = model(input).detach()
        
        output = F.softmax(output, dim=1) * 255
        output = output.squeeze().to('cpu').numpy()

        return output.transpose(1, 2, 0).astype(np.uint8)


    def _multi_scaler(self, img: 'np.ndarray[np.uint8] (H, W, 3)', scale: List[float]) -> 'np.ndarray[np.uint8] (H, W, 3)':

        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img
