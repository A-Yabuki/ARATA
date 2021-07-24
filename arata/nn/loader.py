# coding: utf-8

import copy
import cv2
import glob
import io
import os
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image

from arata.common import image_tools
from arata.common import path_utils



class DataLoader():

    def __init__(self, class_dict):
        self._class_dict = class_dict


    def load_data(self,
                  image_root_path,
                  label_root_path,
                  prediction: bool = False, 
                  batch_size: int = 1,
                  oversampling_num: int =0, 
                  add_noise: bool = True, 
                  add_rotate: bool = True, 
                  apply_clahe: bool = True, 
                  use_cutmix: bool = True, 
                  val_ratio: float = 0):

        batch_size = int(batch_size)
        dataset_paths = self._getimages(image_root_path, label_root_path)
        dataset_paths = self._oversample(dataset_paths[0], dataset_paths[1], oversampling_num)

        transform = transforms.Compose([transforms.ToTensor()])
        
        if not prediction:

            for_training, for_val = self._divide_into_2_datasets(dataset_paths, ratio=val_ratio)

            dataset = NNDataset(for_training[0], for_training[1], self._class_dict, transform)
            builder = DatasetBuilder(dataset)
            builder.set_random_transform(p=0.8)
            builder.set_cut_mix(p=0.1)
            builder.set_dark_edge(p=0.1)
            builder.set_random_marker(p=0.1)
            builder.set_random_point_noise(p=0.1)
            builder.set_unsharp_mask(p=0.1)
            builder.set_gaussian_blur(p=0.3, kmin=3, kmax=7, sigmin=5, sigmax=9)
            builder.set_clahe(p=0.1)

            training_dataset = builder.build()
            
            # num_workers... 並列で利用するCPUの数。高速化できる。
            # pin_memory... CPUのページングを無くし、高速化
            trainloader = torch.utils.data.DataLoader(
                            training_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            
            val_dataset = []
            if len(for_val) != 0:
                val_dataset = builder.build()
                valloader = torch.utils.data.DataLoader(
                                val_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) 
        
            return trainloader, valloader


        else:
            
            dataset = NNDataset(dataset_paths[0], dataset_paths[1], self._class_dict, transform)
            builder = DatasetBuilder(dataset)
            eval_dataset = builder.build()

            evalloader = torch.utils.data.DataLoader(eval_dataset,
                    batch_size=1, shuffle=False)
        
            return evalloader


    def _getimages(self, image_root_path, label_root_path):

        image_paths = path_utils.get_image_paths(image_root_path)
        label_paths = path_utils.get_image_paths(label_root_path)
        
        return image_paths, label_paths


    def _oversample(self, image_paths, label_paths, num):
        num = int(num)
        oversampled_image_paths = []
        oversampled_label_paths = []

        # oversampling (not random oversampling)
        if num > 0:

            input_root_color = self._class_dict['root'].color
            for image_path, label_path in zip(image_paths, label_paths):

                oversampled_image_paths.append(image_path)
                oversampled_label_paths.append(label_path)

                label = cv2.imread(label_path, cv2.IMREAD_COLOR)
                h, w, ch = label.shape

                label = image_tools.binarize(label, input_root_color, input_root_color)/255
                root_occupation = np.sum(label) / (h*w*ch)
                if root_occupation > 0.05:
                    for _ in range(num):
                        oversampled_image_paths.append(image_path)
                        oversampled_label_paths.append(label_path)

            return oversampled_image_paths, oversampled_label_paths
        
        else: 
            return image_paths, label_paths


    def _errorchecker(self, a, b):
        a = glob.glob(a+"/*")
        b = glob.glob(b+"/*")

        for i, j in zip(a, b):

            _, ifile = os.path.split(i)
            ifile = ifile.split(".")[0]

            _, jfile = os.path.split(j)
            jfile = jfile.split(".")[0]

            if ifile!=jfile:

                return True

        return False


    def _divide_into_2_datasets(self, dataset, ratio=0):

        def pop_list(l, indices):

            t = copy.deepcopy(l)
            v = []
            
            for j in indices:

                v.append(l[j])
                t.remove(l[j])


            return t, v

        data, label = dataset
        datanum = len(data)
        selector  = np.random.choice(datanum, int(datanum*ratio), replace=False)
        data, valdata = pop_list(data, selector)
        label, vallab = pop_list(label, selector)
        return [[data, label], [valdata, vallab]]



# 自作データセット
class NNDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, label_path, class_dict, transform):

        self._transform = transform
        self._data_path, self._label_path = data_path, label_path
        self._class_dict = class_dict
        self._data = []
        self._labels = []

        self._transforms = []
        self._processes = []
        
        self._idx = 0

        self._initialize()


    # 長さを返すメソッドが必須
    def __len__(self):
        return len(self._data_path)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx == len(self):
            raise StopIteration()

        value = self[self._idx]
        self._idx += 1
        
        return value

    # 一つ一つのデータを返すメソッドも必須
    def __getitem__(self, idx):

        out_data = copy.deepcopy(self._data[idx])
        out_label = copy.deepcopy(self._labels[idx])
        out_data = self._decompress_data(out_data)
        out_label = self._decompress_data(out_label)

        for fun in self._transforms:
            out_data, out_label = fun(out_data, out_label)

        for fun in self._processes:
            out_data = fun(out_data)

        ### Normalize
        out_data = self._normalize(out_data)
       
        h, w = out_label.shape[:2]
        out_class_label = np.zeros((h, w), dtype=np.uint8)

        for class_info in self._class_dict.values():
            tmp = self._make_class(out_label, class_info.color, class_info.index)
            out_class_label += tmp.astype(np.uint8)

        out_data = self._transform(out_data.astype(np.float32))
        out_class_label = torch.tensor(out_class_label, dtype=torch.long)

        return out_data, out_class_label


    def append_process(self, fun: Callable) -> None:
        self._processes.append(fun)

    def append_transform(self, fun: Callable) -> None:
        self._transforms.append(fun)

    def _initialize(self):
        if (len(self._data_path) != len(self._label_path)):
            raise IndexError("The image data is not the same number as the label data.")
        
        self._zip_dataset()

    def _normalize(self, data):
        return data / 255.

    def _make_class(self, label, color, val: int):
        class_info = self._normalize(image_tools.binarize(label, color, color)) * val
        return class_info.astype(np.uint8)

    def _zip_dataset(self):

        for i in range(len(self)):
        
            compressed_img = self._compress_data(cv2.imread(self._data_path[i], cv2.IMREAD_COLOR))
            compressed_lbl = self._compress_data(cv2.imread(self._label_path[i], cv2.IMREAD_COLOR))

            self._data.append(compressed_img)
            self._labels.append(compressed_lbl)


    def _compress_data(self, data) -> io.BytesIO:

        compressed_data = io.BytesIO()
        np.savez_compressed(compressed_data, data)

        return compressed_data


    def _decompress_data(self, compressed_data: io.BytesIO):

        compressed_data.seek(0)
        data = np.load(compressed_data)["arr_0"]

        return data


class DatasetBuilder():

    def __init__(self, dataset):
        self._dataset = dataset
        self._cut_mix = image_tools.CutMix()

    def set_random_point_noise(self, p: float):
        fun = self._probability_wrapper(image_tools.RandomNoiseCreator.add_random_point_noise, p)
        self._dataset.append_process(fun)

    def set_unsharp_mask(self, p: float):
        fun = self._probability_wrapper(image_tools.RandomNoiseCreator.apply_random_unsharpmask, p)
        self._dataset.append_process(fun)

    def set_random_marker(self, p: float):
        fun = self._probability_wrapper(image_tools.RandomNoiseCreator.add_random_marker_line, p)
        self._dataset.append_process(fun)

    def set_dark_edge(self, p: float):
        fun = self._probability_wrapper(image_tools.RandomNoiseCreator.make_periphery_darker_randomly, p)
        self._dataset.append_process(fun)

    def set_gaussian_blur(self, p: float, kmin: int, kmax: int, sigmin: int, sigmax:int):
        ksize = np.random.choice((kmin, kmax))
        sigma = np.random.choice((sigmin, sigmax))
        fun = self._probability_wrapper(partial(cv2.GaussianBlur, ksize=(ksize,ksize), sigmaX=sigma, sigmaY=sigma), p)
        self._dataset.append_process(fun)

    def set_clahe(self, p: float):
        fun = self._probability_wrapper(image_tools.ContrastControler.clahe, p)
        self._dataset.append_process(fun)

    def set_random_transform(self, p: float):
        self._dataset.append_transform(image_tools.Transformer.flip_randomly)
        self._dataset.append_transform(image_tools.Transformer.rotate_randomly)

    def set_cut_mix(self, p: float):
        fun = self._probability_wrapper(self._cut_mix.cut_or_mix, p)
        self._dataset.append_transform(fun)

    def build(self) -> NNDataset:
        return self._dataset

    def _probability_wrapper(self, fun: Callable, p_thresh: float) -> Callable:

        def wrap(*args):
            p_val = np.random.uniform(low=0.0, high=1.0)
            if p_val <= p_thresh:
                return fun(*args)
            
            else:
                return args if len(args) > 1 else args[0]

        return wrap