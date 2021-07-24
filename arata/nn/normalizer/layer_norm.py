from torch import nn

class LayerNorm2d(nn.LayerNorm):

    def __init__(self):
        self.layer_norm = None

    def __call__(self, input):
        if (self.layer_norm==None):
            self.layer_norm = nn.LayerNorm(input.size()[1:])

        return self.layer_norm(input)