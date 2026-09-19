# coding: utf-8

from collections import namedtuple
import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .evaluator.evaluator import Accuracy, Jaccard
from .net import functions
from .network.deeplab_v3plus_xception import DeepLabSettings
from .vision import progress_viz

from arata.common.interprocess_communicator import report_progress

class Validator():
    """description of class"""

    def __init__(self, trainer, val_data, out_path):

        self.trainer = trainer
        self.train_data_num = len(trainer.train_data)
        self.val_data = val_data
        self.out_path = out_path
        
        self.loss_shift = []
        self.val_loss_shift = []


        self.val_acc_shift = []
        self.val_mcc_shift = []
        self.val_dice_shift = []
        self.val_iou_shift = []

        self.val_acc_calculator = Accuracy()
        self.val_iou_calculator = Jaccard(num_cl=DeepLabSettings.num_class, eps=1e-8, ignore_label=[2])

        self.csv_columns = [
            "Epoch", "Iteration", "Learning Rate", "Loss", "Loss(Validation)",
            "Accuracy", "Dice Coefficient", "MCC", "Mean IoU"]
        

    def append_loss_value(self, loss):
        self.loss_shift.append(loss)


    def validate(self):

        if (len(self.val_data) == 0):
            return

        val_loss = 0.0
        
        self.val_acc_calculator.clear()
        self.val_iou_calculator.clear()
        
        self.trainer.net.eval()

        with torch.no_grad():

            for j, data in enumerate(self.val_data, 0):                
                        
                inputs, labels = data
                inputs = functions.try_gpu(inputs).requires_grad_(False)
                labels = functions.try_gpu(labels).requires_grad_(False)

                outputs = self.trainer.net(inputs).detach()

                loss =  self.trainer.lossfunc(outputs, labels).detach()

                outputs = outputs.argmax(1).squeeze()
                labels = labels.squeeze()
                self.val_acc_calculator.calc(outputs.to('cpu').numpy(), labels.to('cpu').numpy(), i=self.trainer.class_dict["root"].index)
                self.val_iou_calculator.jaccard_calc(outputs.to('cpu').numpy(), labels.to('cpu').numpy())


                val_loss += loss.data.to('cpu').numpy()

            if self.trainer.epoch % self.trainer.output_img_interval == 0:

                test = outputs.size()
                progress_viz.output_img(inputs.squeeze().to('cpu').numpy().transpose(1, 2, 0), outputs.to('cpu').numpy(),  labels.to('cpu').numpy(), self.trainer.class_dict.values(), '{0}/{1}.png'.format(self.out_path, self.trainer.epoch))

        self.val_loss_shift.append(val_loss/(j+1))
            
        self.val_acc_shift.append(self.val_acc_calculator.accuracy())
        self.val_mcc_shift.append(self.val_acc_calculator.matthews_CC())
        self.val_dice_shift.append(self.val_acc_calculator.dice_coef())
        self.val_iou_shift.append(self.val_iou_calculator.jaccard())
        self.val_acc_calculator.clear()
        self.val_iou_calculator.clear()


    def print_progress(self):

        progress = (self.trainer.epoch * 100) // self.trainer.epochs

        msg = '[%d, %5d] lr: %.4e loss: %.4f valloss: %.4f' % \
            (self.trainer.epoch, (self.trainer.epoch)*(self.train_data_num), 
                self.trainer.scheduler.get_last_lr()[0], self.loss_shift[-1], 
                self.val_loss_shift[-1])

        csv_outpath = self.out_path+'/learning_progress.csv'
        out_header = not os.path.exists(csv_outpath)
        dt = pd.DataFrame(np.array([[
            self.trainer.epoch,
            self.trainer.epoch * self.train_data_num,
            self.trainer.scheduler.get_last_lr()[0],
            self.loss_shift[-1],
            self.val_loss_shift[-1],
            self.val_acc_shift[-1],
            self.val_dice_shift[-1],
            self.val_mcc_shift[-1],
            self.val_iou_shift[-1],
        ]]))

        dt.to_csv(csv_outpath, sep=',', mode='a', header=out_header, index=False)

        report_progress('training', int(progress), msg)

        # print learning progress

        if (self.trainer.output_graph_flag and (self.trainer.epoch % self.trainer.output_graph_interval == 0)):

            progress_viz.graph_viz(self.out_path+'/progress.png', 'loss', ('train', self.loss_shift),('val', self.val_loss_shift))
            progress_viz.graph_viz(self.out_path+'/val_accuracy.png', 'Acc&MCC', ('Acc', self.val_acc_shift),('MCC', self.val_mcc_shift),('DICE', self.val_dice_shift))
            progress_viz.graph_viz(self.out_path+'/val_jaccard.png', 'MeanIoU', ('MeanIoU', self.val_iou_shift))
