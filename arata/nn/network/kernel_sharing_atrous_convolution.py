from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

class KernelSharingAtrousSpatialPyramidPooling(nn.Module):
    """
    Cited from : https://arxiv.org/pdf/1908.09443.pdf

    By sharing kernels in the ASPP layer, the number of required parameters were reduced without loss of accuracy. 
    """    
    def __init__(self, in_ch, out_ch, dilation=(0, 0, 0)):

        super(KernelSharingAtrousSpatialPyramidPooling, self).__init__()
        
        pad_size = [i for i in dilation]
        
            
        self.pyramid1 = nn.Conv2d(in_ch, out_ch, 
                                kernel_size = 1, stride=1, padding=0, bias=False)  

        self.pyramid2d = nn.Conv2d(in_ch, in_ch,
                                kernel_size=3, padding=pad_size[0], dilation=dilation[0], groups=in_ch, bias=False)
        self.pyramid2p = nn.Conv2d(in_ch, out_ch,
                                kernel_size=1, bias=False)

        self.pyramid3d = nn.Conv2d(in_ch, in_ch,
                                kernel_size=3, padding=pad_size[1], dilation=dilation[1], groups=in_ch, bias=False)
        self.pyramid3p = nn.Conv2d(in_ch, out_ch,
                                kernel_size=1, bias=False)

        self.pyramid4d = nn.Conv2d(in_ch, in_ch,
                                kernel_size=3, padding=pad_size[2], dilation=dilation[2], groups=in_ch, bias=False)
        self.pyramid4p = nn.Conv2d(in_ch, out_ch,
                                kernel_size=1, bias=False)

        self.conv_encoder_output = nn.Conv2d((out_ch*4+in_ch), out_ch,
                                kernel_size=1, bias=False)

        self.depthwise = [self.pyramid2d, self.pyramid3d, self.pyramid4d]
        size = self.pyramid2d.weight.size()

        # ASPP sharing one weight 
        self.shared_weights = self.pyramid2d.weight

        for f in self.depthwise:
            f.weight = self.shared_weights

    #@profile
    def forward(self, x):

        h1 = self.pyramid1(x)
        h2 = self.pyramid2p(self.pyramid2d(x))
        h3 = self.pyramid3p(self.pyramid3d(x))
        h4 = self.pyramid4p(self.pyramid4d(x))
        hG = F.avg_pool2d(x, kernel_size=x.size()[2:])

        hG = hG.expand(x.size())

        h = torch.cat((h1, h2, h3, h4, hG), dim=1)

        h = self.conv_encoder_output(h)
        
        return h