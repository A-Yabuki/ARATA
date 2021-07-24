# coding: utf-8

import numpy as np
from typing import List, Tuple, Union

from .l1 import tf
from .skeletonizer import skeletonize

def calc_skeletonlen(src: 'np.ndarray[np.uint8] (H, W, 3)',  
                    color: Union['np.ndarray[np.uint8] (3,)', Tuple[int, int, int]]) -> float:

    r"""
    Skeletonize given colored area and calcurates skeleton length (px).
    
    Args:
        src ('np.ndarray[np.uint8] (H, W, 3)'): image 
        color ('np.ndarray[np.uint8] (3,))': color of the object which will be skeletonized.

    Returns:
        length of thinned line (px).
    """

    bin_src = np.where((src == color).all(axis=2), 255, 0)

    # skeletonize by hilditch algorithm
    skeleton = skeletonize(bin_src)
        
    # length calculation
    length = len(np.where(skeleton==255)[0])

    return length


def calc_area(src: 'np.ndarray[np.uint8] (H, W, 3)',  
            color: Union['np.ndarray[np.uint8] (3,)', Tuple[int, int, int]]) -> float:
        
    r"""
    Calculates area of given color.

    Args:
        src ('np.ndarray[np.uint8] (H, W, 3)'): image 
        color ('np.ndarray[np.uint8] (3,))': color of the object.

    Returns:
        float: calculated area (the num of px in the area)
    """

    bin_src = np.where((src == color).all(axis=2), 255, 0)
        
    area = np.sum(bin_src) 

    return area


def cvt_dpi2mmpd(dpi: float) -> float:

    r"""
    Converts given dot per inch value to mm per dot value.
    """

    if (dpi <= 0):
        raise ValueError("Given dpi is a invalid  value.")

    return 25.4 / dpi


def extract_trend(signal: List[float]):
    return tf.l1(signal, sigma=0.005)
    