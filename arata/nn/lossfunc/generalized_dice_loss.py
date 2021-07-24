#!/usr/bin/python

# -*- encoding: utf-8 -*-


### COPY FROM https://github.com/CoinCheung/pytorch-loss/blob/master/dice_loss.py


import torch

import torch.nn as nn





class GeneralizedSoftDiceLoss(nn.Module):

    def __init__(self,

                 p=1,

                 smooth=1,

                 reduction='mean',

                 weight=None,

                 ignore_lb=255):

        super(GeneralizedSoftDiceLoss, self).__init__()

        self.p = p

        self.smooth = smooth

        self.reduction = reduction

        self.weight = None if weight is None else torch.tensor(weight)

        self.ignore_lb = ignore_lb



    def forward(self, logits, label):

        '''

        args: logits: tensor of shape (N, C, H, W)

        args: label: tensor of shape(N, H, W)

        '''

        # overcome ignored label

        ignore = label.data.cpu() == self.ignore_lb

        label = label.clone()

        label[ignore] = 0

        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)

        ignore = ignore.nonzero()

        _, M = ignore.size()

        a, *b = ignore.chunk(M, dim=1)

        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0

        lb_one_hot = lb_one_hot.detach()



        # compute loss

        probs = torch.sigmoid(logits)

        numer = torch.sum((probs*lb_one_hot), dim=(2, 3))

        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3))

        if not self.weight is None:

            numer = numer * self.weight.view(1, -1)

            denom = denom * self.weight.view(1, -1)

        numer = torch.sum(numer, dim=1)

        denom = torch.sum(denom, dim=1)

        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)



        if self.reduction == 'mean':

            loss = loss.mean()

        return loss