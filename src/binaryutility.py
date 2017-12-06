from __future__ import print_function


import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable

import torch.optim as optim
import torch.nn.functional as F
import random

import torchvision
import os
import torchvision.transforms as transforms
from utils import progress_bar

random.seed(1)

#hard sigmoid functiom
def hard_sigmoid(x):
    return torch.clamp((x+1.)/2., 0, 1)

# takes input from [ -1 , 1 ]
# and performs a binarization using hard sigmoid function 
def binarization(W,H, binary=True):
    if not binary:
        Wb = W
    else:
        Wb = hard_sigmoid(W/H)

        Wb = torch.round(Wb)
    
        Wb[Wb > 0 ] = H
        Wb[Wb <=0 ] = -H
   
    return Wb


class BinaryLayer(nn.Linear):

    #initialize the 
    def __init__(self, input_dim, output_dim, H,binary=True):

        self.H = H
        self.binary = binary
        super(BinaryLayer, self).__init__(input_dim, output_dim)
        if self.binary:
            self.Wb = Variable(torch.randn(self.weight.size()), requires_grad=False)
                
    
    def forward(self, input_weights):

        if self.binary:
            #print("binary forward bin")
            self.Wb = binarization(self.weight, self.H, self.binary)
            backup_weight = self.weight.data
            self.weight.data = self.Wb.data

            #print(self.weight.data, self.Wb)

        out = super(BinaryLayer, self).forward(input_weights)

        if self.binary:
            self.weight.data = backup_weight

        return out


class BinaryConvLayer(nn.Conv2d):

    #initialize the 
    def __init__(self, input_channels, output_channels, kernel_size, H, padding=0, binary=True):

        self.H = H
        self.binary = binary
        super(BinaryConvLayer, self).__init__(input_channels, output_channels, kernel_size)

        if self.binary:
            self.Wb = Variable(torch.randn(self.weight.size()), requires_grad=False)
    
    def forward(self, input_weights):

        if self.binary:
            self.Wb = binarization(self.weight, self.H, self.binary)
            backup_weight = self.weight.data
            self.weight.data = self.Wb.data

        out = super(BinaryConvLayer, self).forward(input_weights)
        
        if self.binary:
            self.weight.data = backup_weight

        return out



class ClippedOptimizer(optim.SGD):
    def __init__(self, params, H, lr,binary, momentum=0, dampening=0,weight_decay=0, nesterov=False):
        super(ClippedOptimizer, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.H = H
        self.binary = binary


    def step(self, closure=None):
        super(ClippedOptimizer, self).step(closure)
        
        '''
        if self.binary:
            #print("binary optim")
            for wt in self.param_groups:
                #print(wt['params'])
                for p in wt['params']:
                    #print(p)
                    if p.grad is None:
                        continue
                    p.data = torch.clamp(p.data, -self.H, self.H)
        '''

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG11', bin = False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                if(bin):
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if(bin):
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),nn.BatchNorm2d(x),nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),nn.BatchNorm2d(x),nn.ReLU(inplace=True)]
                in_channels = x
        if(bin):
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        else:
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)