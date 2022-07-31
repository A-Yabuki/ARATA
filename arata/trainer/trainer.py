# coding-utf-8

import datetime
import os
import shutil

from arata.common.configurations import TrainingConfig
from arata.common.constants import ResourcePathConst, TrainingConfigJsonConst
from arata.common.enum import ErrorType
from arata.common.error_handler import ChainedErrorHandler, DetailedErrorMessages
from arata.common.platform_utils import mkdir_recursive
from arata.common.wrapper_collection import watch_error
from arata.nn.loader import DataLoader
from arata.nn.param_setter import create_trainer
from arata.nn.trainer import Trainer

class Trainer():

    r"""
    Trains the network to be able to recognize roots in images.
    """

    def __init__(self) -> None:
        
        loader = TrainingConfig()
        loader.load()
        self.config = loader.config
        self.class_info = loader.class_info


    def trains(self) -> None:

        r"""
        Trains network parameters.
        """

        train_data, val_data = self._loads_data()
        trainer = create_trainer(self.config, self.class_info, train_data, val_data)

        try:
            trainer.train()

        except:
            ChainedErrorHandler("An error occurred while training the AI", ErrorType.CRITICAL_ERROR)
            return

        mkdir_recursive(ResourcePathConst.LOG_OUTPUT_PATH)

        setting_file_name = os.path.splitext(os.path.split(ResourcePathConst.TRAINING_CONFIG_PATH)[1])[0]
        file_name = "{0}_{1}{2}".format(setting_file_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), '.json')
        shutil.copy2(ResourcePathConst.TRAINING_CONFIG_PATH, os.path.join(ResourcePathConst.LOG_OUTPUT_PATH, file_name))


    @watch_error(
        DetailedErrorMessages.get_critical_error_message("loading training data"), 
        ErrorType.CRITICAL_ERROR)
    def _loads_data(self) -> None:

        r"""
        Loads training data.
        """

        loader = DataLoader(self.class_info.class_dict)

        train_loader, val_loader = loader.load_data(
            self.config[TrainingConfigJsonConst.IMAGE_SOURCE],
            self.config[TrainingConfigJsonConst.LABEL_SOURCE],
            False,
            self.config[TrainingConfigJsonConst.BATCH_SIZE], 
            self.config[TrainingConfigJsonConst.OVERSAMPLING],
            self.config[TrainingConfigJsonConst.ADD_RANDOM_NOISE], 
            self.config[TrainingConfigJsonConst.ADD_FLIP_AND_ROTATION], 
            self.config[TrainingConfigJsonConst.APPLY_CLAHE], 
            self.config[TrainingConfigJsonConst.CUT_MIX], 
            self.config[TrainingConfigJsonConst.VALIDATION_RATIO], 
            )
        return train_loader, val_loader


def train() -> None:

    r"""
    Training 
    """

    TN = Trainer()    
    TN.trains()