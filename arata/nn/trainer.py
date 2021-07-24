# coding: utf-8
from functools import partial
from typing import Callable, Dict

import torch
import torch.cuda
from torch.autograd import Variable

from arata.common.enum import ErrorType
from arata.common.error_handler import ChainedErrorHandler, DisplayErrorMessages
from . import functions
from .loader import NNDataset
from .net import NNClientBase
from .validator import Validator


class Trainer(NNClientBase):

    def __init__(self, 
                 train_data: NNDataset, 
                 val_data: NNDataset, 
                 epochs: int, 
                 class_dict: Dict, 
                 out_path: str):

        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = int(epochs)
        self.class_dict = class_dict
        self.out_path = out_path
        self.optimizer = None
        self.lossfunc = None
        self.scheduler = None
        self.output_graph_flag = True
        self.output_img_flag = True
        self.output_graph_interval = 1
        self.output_img_interval = 1
        self.take_snapshot_flag = True
        self.snapshot_interval = 1


    def set_optimizer(self, opt: Callable) -> None:
        self.optimizer = opt


    def set_loss_function(self, lossfunc: Callable) -> None:
        self.lossfunc = lossfunc


    def set_scheduler(self, schedular: Callable) -> None:
        self.scheduler = schedular


    def set_snapshot(self, take_snapshot_flag: bool = True, snapshot_interval: int = 1) -> None:
        self.take_snapshot_flag = take_snapshot_flag
        self.snapshot_interval = int(snapshot_interval)


    def set_output_graph(self, output_graph_flag: bool, output_interval: int) -> None:
        self.output_graph_flag = output_graph_flag
        self.output_grpah_interval = int(output_interval)


    def set_output_img(self, output_img_flag: bool, output_interval: int) -> None:
        self.output_img_flag = output_img_flag
        self.output_img_interval = int(output_interval)


    def train(self) -> None:

        r"""
        Trains feeded network parameters.
        """

        self.validator = Validator(self, self.val_data, self.out_path)

        self.optimizer = self.optimizer(self.net.parameters())
        self.scheduler = self.scheduler(self.optimizer)

        # This makes network fixed and enhance training speed.
        torch.backends.cudnn.benchmark = True

        for self.epoch in range(1, self.epochs+1):

            self.train_1epoch()
                
            self.validator.validate()
            self.validator.print_progress()

            self.scheduler.step()
            self._output_snapshot()

        torch.save(self.net.state_dict(), self.out_path+'/model.pth')


    def train_1epoch(self):

        running_loss = 0.0
        self.net.train()

        # Automatic Mixed Precision ... FP32をFP16で精度を落とさずメモリ使用量節約
        # scalerを作成し、forward, loss, backpropagation, param update をラップすることで利用。
        scaler = torch.cuda.amp.GradScaler()

        for i, data in enumerate(self.train_data, 0):
                
            try:
                self.optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = Variable(functions.try_gpu(inputs)), functions.try_gpu(Variable(labels))

                # wrap the process until calculating loss.
                with torch.cuda.amp.autocast():

                    outputs = self.net(inputs)
                    loss =  self.lossfunc(outputs, labels)
               
                # AMP Train your model
                #with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    #scaled_loss.backward()

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                scaler.step(self.optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                running_loss += loss.data

            except (MemoryError):
                ChainedErrorHandler(DisplayErrorMessages.MEMORY_OVERFLOW, ErrorType.CRITICAL_ERROR)

            except:
                ChainedErrorHandler("Stopped training by an occured error.", ErrorType.CRITICAL_ERROR)

        self.validator.append_loss_value(running_loss/(i+1))


    def _output_snapshot(self) -> None:

        if (self.take_snapshot_flag and (self.epoch % self.snapshot_interval == 0)):
            
            try:
                torch.save(self.net.state_dict(), self.out_path+'/snapshot%d.pth'% (self.epoch))

            except:
                ChainedErrorHandler(
                    DisplayErrorMessages.OUTPUT_FAILED%("snapshot", self.out_path+'/snapshot%d.pth'% (self.epoch)), 
                    ErrorType.IGNORABLE_ERROR)

