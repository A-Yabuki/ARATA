# coding: utf-8

import cv2
import numpy as np
import os

from typing import Sequence, Tuple

from arata.common.enum import ErrorType
from arata.common.image_tools import ContrastControler, ExifReader, PhaseOnlyCorrelation, ImageConverter
from arata.common.interprocess_communicator import report_progress
from arata.common.assertion import hasinstances, isempty
from arata.common.wrapper_collection import watch_error


class ImagePreprocessor():

    r"""
    Performs preprocessing before analysis.
    """


    def __init__(self):
        
        self._boundary_y = []
        self._boundary_x = []

        
    @watch_error("Preprocessing Error", ErrorType.CRITICAL_ERROR)
    def preprocess(self, src_paths: Sequence[str], dst: str, 
                 dpi: int, auto_dpi: bool, frm: str, clsp: str, apply_poc: bool = True) -> None:

        """ apply clahe & position shift correction """

        if not hasinstances(src_paths, str) or isempty(src_paths):
            raise Exception("Invalid argument was given. [args: src_paths]")

        base_img = ExifReader.restore_original_orientation(src_paths[0])
        
        if auto_dpi:
            _, base_dpi = ExifReader.read_exif(src_paths[0], read_orientation=False, read_dpi=True)
        
        base_img = self._processing(base_img, clsp)
        self._save(base_img, src_paths[0], dst)
        
        for i, src_path in enumerate(src_paths[1:]):
                     
            target_img = ExifReader.restore_original_orientation(src_path)

            if auto_dpi:
                target_img  = ExifReader.rescale(src_path, base_dpi, target_img)

            target_img = self._processing(target_img, clsp)

            if apply_poc:
                result, boundary = PhaseOnlyCorrelation.relocate(target_img, base_img)
                self._boundary_y.append(boundary[0])
                self._boundary_x.append(boundary[1])

            else:
                result = target_img

            self._save(result, src_path, dst)
            #base_img  = result

            report_progress("preprocessing", (i+2) * 100 // len(src_paths), "")


    def find_position_constant_area(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:

        r"""
        Calculates min-max values of the relative distance against the initial position.

        0 > y : moving  upward compared to initial position.
        0 < y : moving downward
        0 > x : moving left
        0 < x : moving right

        Returns:
            y_min (int): y min value 
            y_max (int): y max value
            x_min (int): x min value
            x_max (int): x max value
        """

        relative_x = np.asarray(self._boundary_x, dtype=np.int16)
        relative_y = np.asarray(self._boundary_y, dtype=np.int16)

        absolute_x = np.cumsum(relative_x)
        absolute_y = np.cumsum(relative_y)
        
        x_min = np.min(absolute_x)
        x_max = np.max(absolute_x)

        y_min = np.min(absolute_y)
        y_max = np.max(absolute_y)

        return ((y_min, y_max), (x_min, x_max))


    def _open(self, src_path: str) -> 'np.ndarray[np.uint8]':
        return ExifReader.restore_original_orientation(src_path)


    def _processing(self, img: 'np.ndarray[np.uint8]', clsp: str):
        
        # convert color spce to BGR
        img = ImageConverter.cvtColor2BGR(img, clsp)

        # apply clahe
        img = ContrastControler.clahe(img)

        return img



    def _save(self, img: 'np.ndarray[np.uint8]', img_path: str, dst_dir: str) -> None:

        # output an image
        _, img_name = os.path.split(img_path)
        cv2.imwrite(dst_dir + '/' + img_name, img)