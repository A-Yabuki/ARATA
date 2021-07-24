import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np

"""
Cited from..
"https://gist.github.com/yudai09/c1ae3ef9bbf1333acc20b28189df9968.js"
"""
def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    return mask.scatter_(1, index, ones)


# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class FocalLossWithOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()


class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps


    def forward(self, input, target):
        
        logit = F.softmax(input, dim=1) # softmax = probability of class

        # x.clamp(min, max): x<=min..min,  min<=x<= max..x, max<=x..max を返す
        logit = logit.clamp(self.eps, 1. - self.eps)  # (b, cl, h, w)

        logit_ls = torch.log(logit) # log_softmax

        loss = F.nll_loss(logit_ls, target)#, reduction="mean")

        """ここまでソフトマックスクロスエントロピー"""
       
        # 末尾に一次元増やしたサイズ　original=(b, h, w)なら (b, h, w, 1)
        
        view = target.size() + (1,) # (b, h, w, 1)
        index = target.view(*view).permute((0, 3, 1, 2)) # (b, 1, h, w)

        # gather(dim=1, index=index)...logit's dim = 1 shows class probability
        #                              index shows correct class of each pixel
        # -> gather correct class' probability 
        # bellow calculation weight misrecognized pixel's loss more than well recognized.
        loss = loss * ((1 - logit.gather(dim=1, index=index).squeeze(1)) ** self.gamma) # focal loss (b, h, w)
        #loss = loss * ((logit.gather(dim=1, index=index).squeeze(1)) ** self.gamma)
        return loss.mean() # (1,) 


if __name__ == "__main__":
    device = torch.device("cuda")

    focal_without_onehot = FocalLossWithOutOneHot(gamma=1)
    focal_with_onehot = FocalLossWithOneHot(gamma=1)
    input = torch.Tensor([[0.3, 0.1, 0.1], [0.3, 0.6, 0.001], [0.01, 0.002, 2.3], [0.01, 0.002, 2.3]]).to(device)
    target = torch.Tensor([0, 1, 1, 2]).long().to(device)

    focal_without_onehot(input, target)
    
    # exception will occur when input and target are stored to GPU(s).
    #focal_with_onehot(input, target)
