# coding: utf-8

import copy
import cv2
import numpy as np
import numba

from functools import reduce
from numba import njit

"""
The hilditch algorithm of the skeletonization 
"""

def skeletonize(gray_src: 'np.ndarray (H, W, 3)') -> 'np.ndarray (H, W, 3)':

    dst = copy.deepcopy(gray_src)

    if np.sum(dst) == 0:
        return dst

    dst[dst!=0] = 1
    prev = dst

    while True:

        dst = _skeletonize_hilditch(prev)

        if len(np.where(dst == -1)[0]) == 0:
            break

        prev = dst.clip(min=0)

    return dst*255


@njit('i1(i1[:])', cache=True)
def _evaluate_connection_index(b):
    """
    caliculates connection index
    """
    d = np.array([0]*10)
    n_odd = np.array([1, 3, 5, 7])

    # 0~9のインデックス
    for i in range(10):

        # id 9 は 元のid 1
        j = 1 if i == 9 else i

        d[i] = 1 if np.abs(b[j]) == 1 else 0

    sum_ = 0

    for i in range(4):
        j = n_odd[i]
        sum_ = sum_ + (1-d[j]) - (1-d[j])*(1-d[j+1])*(1-d[j+2])

    return sum_


# just in time compile with nopython mode
@njit(numba.i1[:, :](numba.i1[:, :], numba.i4[:, :]), cache=True)
def _skeletonizing(dst: 'np.ndarray[np.uint8] (H, W)', cnt: np.int) -> 'np.ndarray[np.uint8] (H, W)':
    """
    This function skeletonizes object in an binary image based on the Hilditch method.
    """
    h, w = dst.shape

    #ps = np.zeros((9,), dtype=np.int8)
    for coodinate in cnt:

        x, y = coodinate

        # omit image edge to avoid occuring index error
        if (x == 0) | (x == w-1):
            continue

        if (y == 0) | (y == h-1):
            continue

        p0 = dst[y, x]
        p1 = dst[y-1, x]
        p2 = dst[y-1, x+1]
        p3 = dst[y, x+1]
        p4 = dst[y+1, x+1]
        p5 = dst[y+1, x]
        p6 = dst[y+1, x-1]
        p7 = dst[y, x-1]
        p8 = dst[y-1, x-1]

        ps = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8], dtype=np.int8)
        p_odds = np.array([p1, p3, p5, p7], dtype=np.int8)

        """
        Alternative code 

        #ps = np.ravel(dst[y-1:y+2, x-1:x+2])

        #p_odds = ps[1::2]

        #ps = np.array([ps[4], ps[1], ps[2], ps[5], ps[8], ps[7], ps[6], ps[3], ps[0]], dtype=np.int8)
        
        """
        """
        In following condition, because ps includes p0,
        these condition explained in some web site of Hilditch algorithm
        differ from my description.
        """

        """
        p8 p1 p2
        p7 p0 p3  condition 2...
        p6 p5 p4            if all of p1, p3, p5 & p7 = 1, the point isn't removed
        """

        # condition 2: omit internal point
        if np.sum(1 - np.abs(p_odds)) < 1:
            continue

        # condition 3: omit end point
        if np.sum(np.abs(ps)) <= 2:
            continue

        # condition 4: omit isolate point.  bellow operation is same as ps.clip(min=0), but not supported by njit
        if np.sum(np.minimum(10, np.maximum(ps, 0))) <= 1:
            continue

        # condition 5: omit the point that connectivity will be lost if removed
        if _evaluate_connection_index(ps) != 1:
            continue

        # condition 6: choose one side of the line whose width=2
        sum_ = 0

        for i in range(1, 9):

            if ps[i] != -1:
                sum_ += 1

            else:
                temp = np.copy(ps)
                temp[i] = 0
                if _evaluate_connection_index(temp) == 1:
                    sum_ += 1

        if sum_ != 8:
            continue

        dst[y, x] = -1

    return dst


def _skeletonize_hilditch(src: 'np.ndarray[np.int] (H, W)'):

    """
    To employ just in time compile, 
    this part was separated from skeletonizing loop,
    because numba doesn't support cv2 libraries.
    """

    dst = np.copy(src).astype(np.int8)

    # condition 1: omit background
    cnts, hierarchy = cv2.findContours(src.astype(
        np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    cnts = np.array(
        [np.squeeze(np.asarray(cnt, dtype=np.int32), axis=1) for cnt in cnts])

    # 親を持たない輪郭＝最外輪郭のみのindexを取り出す
    cnts_id = np.where((np.asarray(hierarchy[0], dtype=np.int32))[:, 3] == -1)

    # 最外輪郭のみ取り出す
    cnts = cnts[cnts_id[0]]

    # reduce(function, iterable, initializer) ... loop function(returned_value(or initializer), iterable's element)
    return reduce(_skeletonizing, cnts, dst)
