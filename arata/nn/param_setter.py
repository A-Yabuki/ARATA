from functools import partial
from typing import Callable

import torch

from arata.common.configurations import TrainingConfig, ClassInfoManager
from arata.common.constants import ActivationFuncJsonConst, LossFuncJsonConst, NormalizationMethodJsonConst, OptimizerJsonConst, TrainingConfigJsonConst
from arata.common.enum import ErrorType
from arata.common.error_handler import ChainedErrorHandler, DisplayErrorMessages

from arata.nn.activator.tanhexp import tanhexp
from arata.nn.functions import try_gpu
from arata.nn.optimizer.adabound import AdaBound
from arata.nn.network.deeplab_v3plus_xception import DeepLabSettings
from arata.nn.loader import NNDataset
from arata.nn.predictor import Predictor
from arata.nn.trainer import Trainer


def create_predictor(config: TrainingConfig, class_info: ClassInfoManager) -> Predictor:

    predictor = Predictor()

        # set noramlizer
    normalizer = _get_normalizer(config[TrainingConfigJsonConst.NORMALIZER])
        
    # set activator
    activator = _get_activation_func(config[TrainingConfigJsonConst.ACTIVATOR])

    # construct network
    num_cl = len(class_info.class_dict.keys())
    predictor.construct_network(
        num_cl, 
        config[TrainingConfigJsonConst.MIDDLE_LAYER_NUM],
        config[TrainingConfigJsonConst.MIDDLE_LAYER_SCALE],
        normalizer, 
        activator
        )

    return predictor


def create_trainer(config: TrainingConfig, class_info: ClassInfoManager,
                    training_data: NNDataset, validation_data: NNDataset) -> Trainer:
          
    trainer = Trainer(
        training_data, validation_data, 
        config[TrainingConfigJsonConst.EPOCH_NUM], 
        class_info.class_dict, 
        config[TrainingConfigJsonConst.OUTPUT_DESTINATION])

    # set optimizer
    opt = _get_optimizer(
        config[TrainingConfigJsonConst.OPTIMIZER],
        config[TrainingConfigJsonConst.INITIAL_LEARNING_RATE],
        config[TrainingConfigJsonConst.FINAL_LEARNING_RATE],
        config[TrainingConfigJsonConst.WEIGHT_DECAY]
        )
        
    trainer.set_optimizer(opt)
        
    # set loss function
    class_weight = [x.weight for x in class_info.class_dict.values()]
    lossfunc = _get_loss_func(
        config[TrainingConfigJsonConst.LOSS_FUNC], 
        torch.Tensor(class_weight))

    trainer.set_loss_function(lossfunc)     
        
    # set noramlizer
    normalizer = _get_normalizer(config[TrainingConfigJsonConst.NORMALIZER])
        
    # set activator
    activator = _get_activation_func(config[TrainingConfigJsonConst.ACTIVATOR])

    # construct network
    num_cl = len(class_info.class_dict.keys())
    trainer.construct_network(
        num_cl, 
        config[TrainingConfigJsonConst.MIDDLE_LAYER_NUM],
        config[TrainingConfigJsonConst.MIDDLE_LAYER_SCALE],
        normalizer, 
        activator
        )

    # set schedular
    scheduler = _get_scheduler(
        config[TrainingConfigJsonConst.SCHEDULAR_STEP_SIZE],
        config[TrainingConfigJsonConst.SCHEDULAR_STEP_RATE]) 
    trainer.set_scheduler(scheduler)
        
    # set output ways
    trainer.set_output_graph(
        config[TrainingConfigJsonConst.OUTPUT_GRAPH_LOG], 
        config[TrainingConfigJsonConst.OUTPUT_GRAPH_INTERVAL])
    trainer.set_output_img(
        config[TrainingConfigJsonConst.OUTPUT_IMAGE], 
        config[TrainingConfigJsonConst.OUTPUT_IMAGE_INTERVAL])
    trainer.set_snapshot(
        config[TrainingConfigJsonConst.TAKE_SNAPSHOT], 
        config[TrainingConfigJsonConst.TAKE_SNAPSHOT_INTERVAL])
        
    # initialize network
    if config[TrainingConfigJsonConst.INITIAL_MODEL_PATH] is not None:
        trainer.initialize_network(config[TrainingConfigJsonConst.INITIAL_MODEL_PATH])
        
    return trainer


def _get_optimizer(opt_str: str, init_lr: float, final_lr: float, weight_decay: float) -> Callable:

    if opt_str == OptimizerJsonConst.SGD:
        return partial(torch.optim.SGD, lr=init_lr, nesterov=False, weight_decay=weight_decay)

    elif opt_str == OptimizerJsonConst.NesterovAG:
        return partial(torch.optim.SGD, lr=init_lr, momentum=0.8, nesterov=True, weight_decay=weight_decay)

    elif opt_str == OptimizerJsonConst.AdaBound:
        return partial(AdaBound, lr=init_lr, final_lr=final_lr, betas=(0.9, 0.99), gamma=1e-3, weight_decay=weight_decay)

    else:
        ChainedErrorHandler(DisplayErrorMessages.INVALID_INPUT_VALUE%("opt", opt_str), ErrorType.INVALID_INPUT_VALUE_ERROR)


def _get_scheduler(step_size: int, step_rate: float) -> Callable:
    return partial(torch.optim.lr_scheduler.StepLR, step_size=int(step_size), gamma=step_rate, last_epoch=-1)


def _get_activation_func(act: str) -> Callable:

    if act == ActivationFuncJsonConst.ReLU:
        return torch.nn.functional.relu

    elif act == ActivationFuncJsonConst.LeakyReLU:
        return partial(torch.nn.functional.leaky_relu, negative_slope=0.2)

    elif act == ActivationFuncJsonConst.TanhExp:
        return tanhexp

    else:
        ChainedErrorHandler(DisplayErrorMessages%("act", act), ErrorType.INVALID_INPUT_VALUE_ERROR)


def _get_normalizer(norm: str) -> Callable:

    if norm == NormalizationMethodJsonConst.Batch:
        return torch.nn.BatchNorm2d

    elif norm == NormalizationMethodJsonConst.Layer:
        return partial(torch.nn.GroupNorm, 1)

    elif norm == NormalizationMethodJsonConst.Instance:
        return torch.nn.InstanceNorm2d

    else:
        ChainedErrorHandler(DisplayErrorMessages.INVALID_INPUT_VALUE%("norm", norm), ErrorType.INVALID_INPUT_VALUE_ERROR)


def _get_loss_func(loss_func: str, class_weight) -> Callable:

    if loss_func == LossFuncJsonConst.CrossEntropy:
        return torch.nn.CrossEntropyLoss()

    elif loss_func == LossFuncJsonConst.WeightedCE:
        weight = try_gpu(class_weight)
        return torch.nn.CrossEntropyLoss(weight)

    elif loss_func == LossFuncJsonConst.FocalCE:
        return FocalLossWithOutOneHot(gamma=2)

    elif loss_func == LossFuncJsonConst.DiceLoss:
        return GeneralizedSoftDiceLoss

    else:
        ChainedErrorHandler(DisplayErrorMessages.INVALID_INPUT_VALUE%("loss func", loss_func), ErrorType.INVALID_INPUT_VALUE_ERROR)