from torch import nn
import torch.nn.init as init_w


class Initial_weight():
    
    def __init__(self, init_weight=None, nonlinearity='relu', param=None):

        self.init_weight = init_weight
        self.nonlinearity = nonlinearity
        self.gain = init_w.calculate_gain(nonlinearity, param=param)
        self.counter = 0
    
    def w_henorm(self, m):
        if type(m) == nn.Conv2d:
            if self.nonlinearity=='leaky_relu':
                a = 0.2

            else:
                a = 0
            init_w.kaiming_normal_(m.weight, a=a, mode='fan_in', nonlinearity=self.nonlinearity)

    def w_norm(self, m):
        if (type(m) == nn.Conv2d) | (type(m) == nn.ConvTranspose2d):
            init_w.normal_(m.weight, mean=0.0, std=1.0)

    def w_xnorm(self, m):
        if type(m) == nn.Conv2d:
            init_w.xavier_normal_(m.weight, gain=self.gain)

    def w_pretrained(self, m):
        
        if type(m) == nn.Conv2d:
            if (self.counter <= 13):
                m.weight.data = list(self.init_weight.values())[self.counter]
            
            elif (self.counter == 22):
                m.weight.data = list(self.init_weight.values())[-1]
                
            self.counter += 1
        
        else:

            self.w_henorm(m)
        """
        #print(self.init_weight.keys())

        elem = list(self.init_weight.items())
        conv_l = [j for i, j in elem if (i!='6.deconv.weight')&(i.endswith('weight'))]
        num = len(conv_l)
        #temp = [i for i, j in elem if (i!='6.deconv.weight')&(i.endswith('weight'))]
        if (type(m)==nn.Conv2d) | (type(m)==nn.ConvTranspose2d) | (type(m)==nn.BatchNorm2d):
            if num-1 > self.counter:
                #print(type(m),  temp[self.counter], m.weight.data.size(), conv_l[self.counter].size())
                m.weight.data = conv_l[self.counter]    

            else:
                self.w_henorm(m)  

            self.counter += 1
        """
    @staticmethod
    def freeze_weight(model):
        for param in model.parameters():
            param.requires_grad = False

        return model

            