# coding: utf-8

import glob
import os
from typing import List, Sequence
from .constants import ImageExtensionConst
from .sort_utils import numerical_sort


def get_file_names(paths: Sequence[str]) -> List[str]:
    return [os.path.split(i)[1] for i in paths]


def get_image_paths(folder_path: str, img_ext: List[str] = None) -> List[str]:
    
    r""" 
    Gets sorted image paths of given extension in given path of a folder. 
    
    Args:
        folder_path (str): Target folder path
        img_ext (List[str]): Target image extentions. If None is given, returns all image file paths.

    Returns:
        img_paths: Collected image paths sorted by their numeric numbers.
    """

    img_paths = []
    if img_ext == None :

        # collect all paths having any image extensions in the target folder
        image_extentions = []

        [image_extentions.extend(i) for i in ImageExtensionConst.IMAGE_EXTENSIONS]
        
        [img_paths.extend(glob.glob('{}/*.{}'.format(folder_path, i)))
            for i in image_extentions]

    else:
        [img_paths.extend(glob.glob('{}/*.{}'.format(folder_path, i)))
                for i in img_ext]

    return sorted(img_paths, key=numerical_sort)