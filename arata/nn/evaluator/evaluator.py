import csv
import cv2
import glob
import numpy as np
import os

import torch
import torch.nn.functional as F

"""
    p: positive, n: negative
    pp: predictive positive
    pn: predictive negative

    up: union of positive (pvpp)
    un: union of negative (nvpn)
    tp: true positive (p^pp)
    fp: false positive (n^pp)
    tn: true negative (n^pn)
    fn: false negative (p^pn)

    tpr: true positive rate (tp/p = tp/(tp+fn)) (recall)
    fpr: false positive rate (fp/n = fp/(fp+tn))
    tnr: true negative rate (tn/n = tn/(tn+fp))
    fnr: false negative rate (fn/p  fn/(fn+tp))
    ppr: positive predictive rate (tp/pp) (precision)
    npr: negative predictive rate (tn/pn)

    Acc: (tp+tn) / (tp+tn+fp+fn)
    Mean IoU: 1/2(tp/up + tn/un)
    F1: 2*(ppr*tpr)/(tpr+ppr)
    MCC: matthews correlation coefficient (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
"""




class Jaccard():

    def __init__(self, num_cl, ignore_label=None, eps=1e-8):

        self.miou = 0
        self.called = 0
        self.num_cl = num_cl
        self.ignore_label = ignore_label
        self.eps = eps

        if type(ignore_label) == list:
            self.num = num_cl - len(ignore_label)
        
        else: self.num = num_cl

    def jaccard_calc(self, y, t):

        mul_ = 0
        sum_ = 0


        for i in range(self.num_cl):

            if i in self.ignore_label:
                continue

            temp_y = np.where(y==i, 1, 0)
            temp_t = np.where(t==i, 1, 0)

            mul_ = np.sum(temp_y * temp_t)
            sum_ = np.sum(temp_y + temp_t) - mul_ + self.eps

            self.miou += mul_/sum_
        
        self.called += 1


    def jaccard(self):
        return np.round(self.miou/(self.num*self.called), decimals=4)

    def clear(self):
        self.miou = 0
        self.called = 0


class Accuracy():

    def __init__(self):
        self.p = 0
        self.n = 0
        self.pp = 0
        self.pn = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.up = 0
        self.un = 0

        self.called_num = 0

    def calc(self, y, t, i):
        
        y = np.where((y==i), 1, 0)
        t = np.where((t==i), 1, 0)


        p = np.sum(t)
        n =  np.size(t) - p
        pp = np.sum(y)
        pn = np.sum(1-y)
        
        tp = np.sum(y*t)
        fp = np.sum(y*(1-t))
        
        tn = np.sum((1-y)*(1-t))
        fn = np.sum((1-y)*t)

        self.p  += p
        self.n  += n
        self.pp += pp
        self.pn += pn
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        self.up += p + pp - tp
        self.un += n + pn - tn
        
        self.called_num += 1 

    def accuracy(self):
        return (self.tp+self.tn)/(self.p + self.n)

    def recall(self):
        return (self.tp)/(self.p)

    def precision(self):
        return (self.tp)/(self.pp)

    def area_ratio(self):
        return (self.pp)/(self.p)

    def dice_coef(self):
        return self.tp / (self.tp + (self.fp + self.fn) * (1 / 2))

    def matthews_CC(self):

        numerator = ((self.tp/self.called_num)*(self.tn/self.called_num) - (self.fp/self.called_num)*(self.fn/self.called_num))
        denominator = np.sqrt((self.tp+self.fp)/self.called_num)*np.sqrt((self.tp+self.fn)/self.called_num)*\
            np.sqrt((self.tn+self.fp)/self.called_num)*np.sqrt((self.tn+self.fn)/self.called_num)

        mcc = numerator/(denominator+1e-8)
        return mcc

    def clear(self):
        self.p = 0
        self.n = 0
        self.pp = 0
        self.pn = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.up = 0
        self.un = 0
        self.called_num = 0