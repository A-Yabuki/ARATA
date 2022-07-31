import cv2
import numpy as np
import os

from typing import List

from arata.common.enum import ErrorType
from arata.common.interprocess_communicator import report_progress
from arata.common.wrapper_collection import watch_error


class Postprocessor():

    def __init__(self, img_paths: List[str], dst_folder_name: str) -> None:

        self.img_paths = img_paths
        img_file_name_format = dst_folder_name + "/{}.png"
        self.formatter = img_file_name_format.format


    def run(self) -> None:

        for idx, img_path in enumerate(self.img_paths, start=1):

            cv2.imwrite(
                self.formatter(os.path.splitext(os.path.split(img_path)[1])[0]), 
                self._postprocess(cv2.imread(img_path, cv2.IMREAD_COLOR))
                )

            report_progress("postprocessing", idx * 100 // len(self.img_paths), "")
        


    def _postprocess(self, src: 'np.ndarray[np.uint8] (H, W, 3)') -> 'np.ndarray[np.uint8] (H, W, 3)':

        if (not isinstance(src, np.ndarray)) or src.ndim != 3:
            raise ValueError("Invalid arguments were given to _postprocess")

        out = np.zeros_like(src, dtype=np.uint8)
        colors = np.unique(src.reshape(-1, src.shape[-1]), axis=0)

        for color in colors:

            gray = np.where((src == color).all(axis=2), 255, 0)
            res = self._remove_small_particles(gray, particle_size=50)
            out[np.where(res == 255)] = color
      
        return out


    
    def _remove_small_particles(self, bin_src: 'np.ndarray[np.uint8] (H, W)', particle_size: int = 50) -> 'np.ndarray[np.uint8] (H, W)':

        dst = np.copy(bin_src.astype(np.uint8))
        cnts, hierarchy = cv2.findContours(bin_src.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if hierarchy is None:
            return bin_src
    
        small_cnts = []
        small_cnts_append = small_cnts.append

        for i, floor_ in enumerate(hierarchy[0]):
            if floor_[3] == -1:

                if len(cnts[i]) <= particle_size :
                    small_cnts_append(cnts[i])
  
        cv2.drawContours(dst, small_cnts, -1, 0, -1)


        return dst