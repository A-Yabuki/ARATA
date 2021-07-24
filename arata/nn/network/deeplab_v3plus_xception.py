from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch

#from pytorch_memlab import profile

#from crfseg import CRF
from .kernel_sharing_atrous_convolution import KernelSharingAtrousSpatialPyramidPooling
from .nn_base import NNBase
from ..activator.tanhexp import *

class DeepLabSettings():
    
    """
    This class defines common settings of the whole network. 
    """

    # common activator function
    activator = tanhexp
    normalizer = partial(nn.GroupNorm, 1)
    num_middle_layer = 16
    num_class = 2
    middle_layer_scale = 16


class DeepLabv3plusXception(NNBase, nn.ModuleList):
    
    def __init__(self):
        
        super(DeepLabv3plusXception, self).__init__([
        EntryFlow(3, 32, 728, 48),
        MiddleFlow(728, layer=DeepLabSettings.num_middle_layer),
        ExitFlow(728, 1024, 1536, 2048, scale=1),
        #AtrusSpatialPyramidPooling(2048, 256, dilation=(6, 12, 18)),
        KernelSharingAtrousSpatialPyramidPooling(2048, 256, dilation=(6, 12, 18)),
        PixelShuffler(256, 256, scale=4),
        #Upsampler(256, 256, scale=4),
        AdditionalConv(304, 256),
        #Upsampler(256, DeepLabSettings.num_class, scale=4),
        PixelShuffler(256, DeepLabSettings.num_class, scale=DeepLabSettings.middle_layer_scale // 4),
        #CRF(n_spatial_dims=2, filter_size=11, n_iter=5, requires_grad=True,
        #         returns='logits', smoothness_weight=1, smoothness_theta=1)
        ])

    #@profile
    def forward(self, x):
        
        #b, _, h, w = x.size()
        
        x1, branch = self[0](x)
        x1 = self[1](x1)
        x1 = self[2](x1)
        x1 = self[3](x1)
        x1 = self[4](x1)

        # reshape tensor not to cause size difference to input
        #x1 = x1.resize_((b, -1, int(h/4), int(w/4)))
        _, _, h, w = branch.size()
        x1 = x1[:, :, :h, :w]

        x1 = torch.cat((x1, branch), dim=1)
        x1 = self[5](x1)
        x1 = self[6](x1)

        #x1 = x1.resize_((b, -1, h, w))

        return x1


class EntryFlow(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch, branch_size):


        super(EntryFlow, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, mid_ch, 
                                kernel_size=3, stride=2, padding=1, 
                                bias=False)

        layer2_stride = 2 if DeepLabSettings.middle_layer_scale == 32 else 1

        self.conv2 = nn.Conv2d(mid_ch, mid_ch*2, 
                                kernel_size=3, stride=layer2_stride, padding=1,
                                bias=False)

        flow_a_stride = 1 if DeepLabSettings.middle_layer_scale == 8 else 2
        self.flow_a = SeparableBlock(mid_ch*2, mid_ch*4, scale=flow_a_stride, skip_conv=True)
            
        self.flow_b = SeparableBlock(mid_ch*4, mid_ch*8, scale=2, branch=True)
        
        self.flow_c = SeparableBlock(mid_ch*8, out_ch, scale=2, skip_conv=True)
            
        self.conv_to_decoder = nn.Conv2d(mid_ch*8, branch_size,
                                 kernel_size=1, stride=1, padding=0,
                                 bias=False)


    #@profile                            
    def forward(self, x):
        
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.flow_a(h)
        h, branch = self.flow_b(h)
        h = self.flow_c(h)
        branch = self.conv_to_decoder(branch)

        return h, branch



class MiddleFlow(nn.Module):

    def __init__(self, ch, layer=16):

        super(MiddleFlow, self).__init__()

        self.choices = nn.ModuleDict({})

        for i in range(1, layer):
            self.choices.update([['middle{}'.format(i),
                     SeparableBlock(ch, ch, skip_conv=False)]])
               
        self.layer = layer

    #@profile
    def forward(self, x):

        for i in range(1, self.layer):
            h = self.choices['middle{}'.format(i)](x)

        return h




class ExitFlow(nn.Module):
    
    def __init__(self, in_ch, mid1_size, mid2_size, out_ch, scale=1):

        super(ExitFlow, self).__init__()

        self.choices = nn.ModuleDict([['a', SeparableBlock(in_ch, mid1_size, scale=scale, skip_conv=True)],
                                      ['b1', SeparableConv(mid1_size, mid2_size)],
                                      ['b2', SeparableConv(mid2_size, mid2_size)],
                                      ['b3', SeparableConv(mid2_size, out_ch)]])
    #@profile
    def forward(self, x):
        
        h = self.choices['b3'](self.choices['b2'](self.choices['b1'](self.choices['a'](x))))

        return h



class SeparableBlock(nn.Module):

    def __init__(self, in_ch, out_ch, scale=1, skip_conv=True, branch=False):

        self.skip_conv = skip_conv
        self.branch = branch
    
        super(SeparableBlock, self).__init__()

        self.conv1d = nn.Conv2d(in_ch, in_ch,
                                kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.conv1p = nn.Conv2d(in_ch, out_ch, 
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2d = nn.Conv2d(out_ch, out_ch,
                                kernel_size=3, stride=1, padding=1, groups=out_ch, bias=False)
        self.conv2p = nn.Conv2d(out_ch, out_ch, 
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3d = nn.Conv2d(out_ch, out_ch,
                                kernel_size=3, stride=scale, padding=1, groups=out_ch, bias=False)
        self.conv3p = nn.Conv2d(out_ch, out_ch, 
                                kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = DeepLabSettings.normalizer(in_ch)            
        self.bn2 = DeepLabSettings.normalizer(out_ch)
        self.bn3 = DeepLabSettings.normalizer(out_ch)


        if self.skip_conv:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 
                                kernel_size=1, stride=scale, padding=0, bias=False)
                                

                    
    #@profile
    def forward(self, x):
        
        h = self.conv1p(DeepLabSettings.activator(self.bn1(self.conv1d(x))))

        if self.branch:
            h_branch = self.conv2p(DeepLabSettings.activator(self.bn2(self.conv2d(h))))

            h = self.conv3p(DeepLabSettings.activator(self.bn3(self.conv3d(h_branch))))

            if self.skip_conv:
                x = self.conv_shortcut(x)

            return h + x, h_branch

        else:
            h = self.conv2p(DeepLabSettings.activator(self.bn2(self.conv2d(h))))
            h = self.conv3p(DeepLabSettings.activator(self.bn3(self.conv3d(h))))

            if self.skip_conv:
                x = self.conv_shortcut(x)

            return h + x


class SeparableConv(nn.Module):
    
    def __init__(self, in_ch, out_ch):

        super(SeparableConv, self).__init__()
      
        self.convd = nn.Conv2d(in_ch, in_ch,
                                kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.convp = nn.Conv2d(in_ch, out_ch, 
                                kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn = DeepLabSettings.normalizer(in_ch)

    #@profile
    def forward(self, x):
        
        h = self.convd(x)
        h = self.convp(DeepLabSettings.activator(self.bn(self.convd(x))))

        return h


class AtrusSpatialPyramidPooling(nn.Module):
    
    def __init__(self, in_ch, out_ch, dilation=(0, 0, 0)):

        super(AtrusSpatialPyramidPooling, self).__init__()
        
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


class AdditionalConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(AdditionalConv, self).__init__()

        # output_stride / 4
        self.separableConv1 = SeparableConv(in_ch, out_ch)
        self.separableConv2 = SeparableConv(out_ch, out_ch)
    #@profile
    def forward(self, x):

        h = self.separableConv1(x)
        h = self.separableConv2(h)
        return h


class PixelShuffler(nn.Module):

    def __init__(self, in_ch, out_ch, scale):
        
        super(PixelShuffler, self).__init__()

        self.convp = nn.Conv2d(in_ch, out_ch*(scale**2), 
                                kernel_size=1, stride=1, padding=0, bias=False)

        self.pixshuf = nn.modules.PixelShuffle(scale)

    #@profile
    def forward(self, x):

        h = self.pixshuf(self.convp(x))

        return h


class Upsampler(nn.Module):

    def __init__(self, in_ch, out_ch, scale):
        
        super(Upsampler, self).__init__()

        self.scale = scale

        self.deconv = nn.ConvTranspose2d(in_ch, out_ch,
                        kernel_size=scale, stride=scale, groups=1, bias=False)


    def forward(self, x):

        h = self.deconv(x)
        return h