# coding: utf-8
import json
import os
from collections import namedtuple
from datetime import datetime as dt
from enum import IntEnum
from typing import Dict, List, NamedTuple, Tuple
from types import MappingProxyType

from .constants import AnalysisConfigJsonConst, ImageExtensionConst, JsonItem, ResourcePathConst, TrainingConfigJsonConst
from .enum import ErrorType
from .error_handler import ChainedErrorHandler
from .singleton import Singleton


class ClassInfoManager():

    class ColorPicker():

        r"""
        Converts a color name to corresponding BGR color value
        """

        COLOR = {
            "black": (0, 0, 0),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "aqua": (255, 255, 0), 
            "pink": (255, 0, 255),
            "yellow": (0, 255, 255),
            "white": (255, 255, 255)
            }

        def __init__(self):
            pass

        @classmethod
        def str2color(cls, color_name: str) -> Tuple[int, int, int]:

            r"""
            Gets BGR value of the specified color

            Args:
                color_name (str): color name

            Returns:
                BGR value of the specified color

            """

            if (color_name in cls.COLOR):
                color = cls.COLOR[color_name]

            else:
                color = cls.COLOR["black"]

            return color


    def __init__(self)-> None:

        self.class_info = namedtuple(
            "ClassInfo", 
            ["index", "name", "color", "weight", "calc_flag"])
        
        self._class_dict = {}
        self.index = 0


    def add_class(self, object_name: str, color_str: str,  weight: int, calc_flag: bool) -> None:
        
        r"""
        Assigns a color to each class
        """

        color = self.ColorPicker.str2color(color_str)

        class_info = self.class_info(self.index, object_name, color, weight, calc_flag)
        self.index += 1

        # if same name object doesn't exist, it adds to the dictionary.
        if (not object_name in self._class_dict):
            self._class_dict[object_name] = class_info

        else:
            ChainedErrorHandler(
                "Same class name already exists. Please change the class name.", 
                ErrorType.INVALID_INPUT_VALUE_ERROR)

    @property
    def class_dict(self) -> Dict[str, NamedTuple]:
        return MappingProxyType(self._class_dict)


class JsonLoader(Singleton):

    r"""
    Load a json file.
    """

    def __init__(self) -> None:
        self.config = dict()
   

    def load(self, file_path: str, key_list: List[str]) -> None:
        
        r"""
        Loads a json setting file

        Args:
            file_path (str): json file path
            key_list (list(str)): json key list 
        """

        ext = os.path.splitext(file_path)[1]
        if (not os.path.exists(file_path) or ext != ".json"):
            ChainedErrorHandler(
                str.format("Json file is not exists. path: {0}", file_path), 
                ErrorType.INVALID_INPUT_VALUE_ERROR)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_load = json.load(f)

                for key in key_list:

                    if (not key in config_load):
                        continue

                    tmp = config_load[key][JsonItem.VALUE]

                    if (config_load[key][JsonItem.IS_BOOL]):
                        tmp = True if tmp == "True" else False
                
                    elif (config_load[key][JsonItem.IS_NUMERIC]):
                        if (tmp == None) or (tmp == ''):
                            tmp = 0
                        else:
                            tmp = float(tmp)
                
                    self.config[key] = tmp

        except:
            ChainedErrorHandler(
                "Failed to read json file.",
                ErrorType.CRITICAL_ERROR)


class TrainingConfig(JsonLoader):

    r"""
    Gets and stores current analysis configurations.
    """

    def __init__(self) -> None:
        super().__init__()
        self._class_info = ClassInfoManager()


    def load(self):     
        
        r"""
        Loads training configurations.
        """

        super().load(ResourcePathConst.TRAINING_CONFIG_PATH, TrainingConfigJsonConst.KEY_LIST)

        self.class_info.add_class(
            'background', 
            self.config[TrainingConfigJsonConst.CLASS1_COLOR], 
            self.config[TrainingConfigJsonConst.CLASS1_WEIGHT],
            self.config[TrainingConfigJsonConst.CLASS1_IGNORE] 
            )

        self.class_info.add_class(
            'root', 
            self.config[TrainingConfigJsonConst.CLASS2_COLOR], 
            self.config[TrainingConfigJsonConst.CLASS2_WEIGHT],
            self.config[TrainingConfigJsonConst.CLASS2_IGNORE] 
            )

        self.class_info.add_class(
            'others1', 
            self.config[TrainingConfigJsonConst.CLASS3_COLOR], 
            self.config[TrainingConfigJsonConst.CLASS3_WEIGHT],
            self.config[TrainingConfigJsonConst.CLASS3_IGNORE] 
            )

        self.class_info.add_class(
            'others2', 
            self.config[TrainingConfigJsonConst.CLASS4_COLOR], 
            self.config[TrainingConfigJsonConst.CLASS4_WEIGHT],
            self.config[TrainingConfigJsonConst.CLASS4_IGNORE] 
            )

        self.class_info.add_class(
            'others3', 
            self.config[TrainingConfigJsonConst.CLASS5_COLOR], 
            self.config[TrainingConfigJsonConst.CLASS5_WEIGHT],
            self.config[TrainingConfigJsonConst.CLASS5_IGNORE] 
            )


    @property
    def class_info(self) -> ClassInfoManager:
        return self._class_info


class AnalysisConfig(JsonLoader):

    r"""
    Gets and stores current analysis configurations.
    """

    class PathEnum(IntEnum):
        SAVE_LOC = 0
        ROOT = 1
        MAIN = 2
        PROB = 3
        POST = 4
        DIFF = 5
        INCR = 6
        DECR = 7
        CALC = 8
        LOG = 9
        TABLE = 10


    def __init__(self) -> None:
        super().__init__()

        self._path_dict = {}


    def load(self) -> None:

        r"""
        Loads analysis configs
        """

        super().load(ResourcePathConst.ANALYSIS_CONFIG_PATH, AnalysisConfigJsonConst.KEY_LIST)

        if "JPEG" in self.config[AnalysisConfigJsonConst.IMAGE_FORMAT]:
            self.config[AnalysisConfigJsonConst.IMAGE_FORMAT] = \
                ImageExtensionConst.JPG_EXTENSIONS

        elif "PNG" in self.config[AnalysisConfigJsonConst.IMAGE_FORMAT]:
            self.config[AnalysisConfigJsonConst.IMAGE_FORMAT] = \
                ImageExtensionConst.PNG_EXTENSIONS

        elif "BMP" in self.config[AnalysisConfigJsonConst.IMAGE_FORMAT]:
            self.config[AnalysisConfigJsonConst.IMAGE_FORMAT] = \
                ImageExtensionConst.BMP_EXTENSIONS
        
        elif "TIF" in self.config[AnalysisConfigJsonConst.IMAGE_FORMAT]:
            self.config[AnalysisConfigJsonConst.IMAGE_FORMAT] = \
                ImageExtensionConst.TIF_EXTENSIONS

        else:
            self.config[AnalysisConfigJsonConst.IMAGE_FORMAT] = None


    def make_folders(self) -> None: 

        r"""
        Creates folders to save result.

        ----------------------------------------
        Current directory configuration
        ----------------------------------------

        SAVE LOC (Destination folder designated by user.)
        \
        |-- ROOT (Source folder's name) 
            \
            |-- MAIN (Time at which analysis was started)
                \
                |-- PROB (probability: the file to save images showing the probability of which object each pixel belongs to) 
                |
                |-- POST (postprocess: the file to save images showing which object each pixel belongs to)
                |
                |-- DIFF (difference: the file to save difference images showing increase or decrease of object amount through transition)
                    \
                    |-- INCR (increment:)
                    |   
                    |-- DECR (decrement:)
                |
                |-- CALC
        """

        self._set_folder_paths()

        save_loc = self.config[AnalysisConfigJsonConst.SAVE_LOCATION]
        if (not os.path.exists(save_loc)):
            try:
                os.mkdir(save_loc)

            except:
                error_msg = str.format("Failed to create folders to save. [invalid path: {0}]", save_loc)
                ChainedErrorHandler(error_msg, ErrorType.CRITICAL_ERROR)

        # Create folders

        for folder in self._path_dict.values():
            if not os.path.exists(folder):
                try:
                    os.mkdir(folder)

                except:
                    error_msg = str.format("Failed to create folders to save. [invalid path: {0}]", folder)
                    ChainedErrorHandler(error_msg, ErrorType.CRITICAL_ERROR)
    
        self._set_file_paths()


    def _set_folder_paths(self):

        save_loc = self.config[AnalysisConfigJsonConst.SAVE_LOCATION]
        self._path_dict[self.PathEnum.SAVE_LOC] = save_loc        
        self._path_dict[self.PathEnum.ROOT] = \
            os.path.join(self._path_dict[self.PathEnum.SAVE_LOC], os.path.split(self.config[AnalysisConfigJsonConst.SOURCE_LOCATION])[1])
        self._path_dict[self.PathEnum.MAIN] =  os.path.join(self._path_dict[self.PathEnum.ROOT], dt.now().strftime('%Y%m%d_%H%M%S')) 
        self._path_dict[self.PathEnum.PROB] = os.path.join(self._path_dict[self.PathEnum.MAIN], 'probability')
        self._path_dict[self.PathEnum.POST] = os.path.join(self._path_dict[self.PathEnum.MAIN], 'postprocess')
        self._path_dict[self.PathEnum.DIFF] = os.path.join(self._path_dict[self.PathEnum.MAIN], 'difference')
        self._path_dict[self.PathEnum.INCR] = os.path.join(self._path_dict[self.PathEnum.DIFF], 'increment')
        self._path_dict[self.PathEnum.DECR] = os.path.join(self._path_dict[self.PathEnum.DIFF], 'decrement')
        self._path_dict[self.PathEnum.CALC] = os.path.join(self._path_dict[self.PathEnum.MAIN], 'calculated values')


    def _set_file_paths(self):

        self._path_dict[self.PathEnum.LOG] = os.path.join(self._path_dict[self.PathEnum.CALC], 'analysis log.txt')
        self._path_dict[self.PathEnum.TABLE] = os.path.join(self._path_dict[self.PathEnum.CALC],'calcurated values.csv')
        



    @property
    def result_root_path(self) -> str:
        return self._path_dict[self.PathEnum.ROOT] 


    @property
    def result_main_path(self) -> str:
        return self._path_dict[self.PathEnum.MAIN]


    @property
    def result_probability_path(self) -> str:
        return self._path_dict[self.PathEnum.PROB]


    @property
    def result_postprocess_path(self) -> str:
        return self._path_dict[self.PathEnum.POST]


    @property
    def result_difference_path(self) -> str:
        return self._path_dict[self.PathEnum.DIFF]


    @property
    def result_increment_path(self) -> str:
        return self._path_dict[self.PathEnum.INCR]


    @property
    def result_decrement_path(self) -> str:
        return self._path_dict[self.PathEnum.DECR]


    @property
    def result_calculation_path(self) -> str:
        return self._path_dict[self.PathEnum.CALC]


    @property
    def result_log_path(self) -> str:
        return self._path_dict[self.PathEnum.LOG]


    @property
    def result_table_path(self) -> str:
        return self._path_dict[self.PathEnum.TABLE]