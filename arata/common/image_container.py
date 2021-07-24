# coding: utf-8

from typing import Any, List, Tuple

import cv2

class ImageContainer():

    """Container of loaded images' location"""

    def __init__(self, img_paths: List[str]) -> None:
        self.img_paths = img_paths


    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        return self.img_paths[idx], img


    def __len__(self):
        return len(self.img_paths)