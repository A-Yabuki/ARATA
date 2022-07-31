# coding: utf-8

import cv2
import numpy as np
import os
from itertools import chain
from typing import Dict, List, Tuple

from arata.common.enum import ErrorType
from arata.common.error_handler import ChainedErrorHandler
from arata.common.interprocess_communicator import report_progress
from arata.nn.predictor import Predictor

class RootPainter():

    r"""
    Classifies and paints objects in images.
    """

    def __init__(self) -> None:
        pass


    def run(self,
            img_paths: List[str], 
            dst: str, 
            predictor: Predictor, 
            crop_size: int, 
            dpi: int, 
            multi_scale: bool, 
            roi: Tuple[Tuple[int, int], Tuple[int, int]], 
            class_dict: Dict) -> None:

        r""" 
        Runs root extraction processing.
        Args:
            img_paths (List[str]): image paths
            dst (str): directory path to save results
            predictor (Predictor): predictor
            crop_size (int): crop size
            dpi (int): dpi
            multi_scale (bool): use multi scale or not
            roi (((int, int), (int, int))): region of interest
            class_dict (Dict): dictionary having class informations
        """

        ### cache of functions and methods to acceralate running speed
        
        imread = cv2.imread
        read_file_name = os.path.splitext
        split_path = os.path.split

        ###

        for i, img_path in enumerate(img_paths, start=1):    

            file_name, _ = read_file_name(split_path(img_path)[1])
            img = imread(img_path, cv2.IMREAD_COLOR)
            
            # Add padding, such that the image size can be divided by the crop size

            img, (pad_top, pad_bottom, pad_left, pad_right) = self._pad(img, crop_size)
            hs, ws, cropped = self._crop(img, crop_size)
            flat = list(chain.from_iterable(cropped))

            preds = predictor.predict(flat, multi_scale)
            
            merged = self._merge(preds, hs, ws)

            original_size_merged = merged[pad_top : merged.shape[0] - pad_bottom, pad_left : merged.shape[1] - pad_right]
            colored = self._colorize(original_size_merged, class_dict)

            h, w = colored.shape[:-1]
            min_y = roi[0][0] if roi[0][0] > 0 else 0
            min_x = roi[1][0] if roi[1][0] > 0 else 0
            max_y = h + roi[0][1] if roi[0][1] < 0 else h
            max_x = w + roi[1][1] if roi[1][1] < 0 else w
            
            out = colored[min_y : max_y, min_x : max_x]
            out_path = "{0}/{1}.png".format(dst, file_name)
            success = cv2.imwrite(out_path, out)

            progress = (i) * 100 // len(img_paths)
            report_progress("root tracing", progress, "")


    def _crop(self, img, crop_size):

        # crop images
        crop_size = int(crop_size)
        h, w = img.shape[:2]
            
        hs = [i for i in range(0, h, crop_size)]
        ws = [i for i in range(0, w, crop_size)]
            
        try:

            ret = \
                [
                    [ cropped_img for _ , cropped_img in enumerate(np.split(v_cropped_img, ws[1:], axis=1)) ]
                    for _, v_cropped_img in enumerate(np.split(img, hs[1:], axis=0))
                ]

            return len(hs), len(ws), ret
            
        except:
            ChainedErrorHandler("ARATA failed to crop images.", ErrorType.CRITICAL_ERROR)


    def _pad(self, img, crop_size):
                
        # Add padding, such that the image size can be divided by the crop size

        crop_size = int(crop_size)
        orig_h, orig_w = img.shape[:2]

        remainderH = orig_h % crop_size
        remainderW = orig_w % crop_size
        padH = crop_size - remainderH if remainderH !=0 else 0
        padW = crop_size - remainderW if remainderW !=0 else 0
                
        top = padH // 2
        bottom = padH - top

        left = padW // 2
        right = padW - left

        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                
        return img, (top, bottom, left, right)


    def _merge(self, imgs, hs, ws):

        merged = None
        for h in range(hs):

            tmp_h = None
            for w in range(ws):
                    
                if tmp_h is None:
                    tmp_h = imgs[ws * h + w]
                    
                else:
                    tmp_h = np.concatenate((tmp_h, imgs[ws * h + w]), axis=1)

            if merged is None:
                merged = tmp_h
        
            else:
                merged = np.concatenate((merged, tmp_h), axis=0)

        return merged


    def _colorize(self, pred, color_dict):
        
        h, w = pred.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        classes = pred.argmax(axis=2)

        for value in color_dict.values():
                
            colored[classes == value.index] = value.color
            
        return colored