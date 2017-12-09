from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy

def binarize(W, stochastic=False):
    x = copy.deepcopy(W.data)
    y = torch.clamp(x, -1, 1)
    x = torch.clamp((x+1.0)/2.0, 0, 1)
    x = torch.round(x)
    #     print(x),"HELLLLLLLLLLO",y
    x[x==1] = 1
    x[x==0] = -1
    return x,y

class NewBinaryLayer(nn.Linear):
    #initialize the Binary Layer where weights are binarized
    def __init__(self, input_dim, output_dim, verbose=False):
        self.verbose = verbose
        super(NewBinaryLayer, self).__init__(input_dim, output_dim)
        
        
    def forward(self, x):
        if(self.verbose):
            print("Weights,bias in forward prop before binarization")
            print(self.weight.data)
            print(self.bias.data)
        
        self.new_weight,clipped_wt_data = binarize(self.weight)
        
        if(self.verbose):
            print(self.weight.data)
            print(self.new_weight)
        self.weight.data = clipped_wt_data
        backup_weight = self.weight.data
        self.weight.data = self.new_weight
        if(self.verbose):
            print("inputs")
            print(x.data)

            print("Weights,bias in forward prop after binarization")
            print(self.weight.data)
            print(self.bias.data)
        
        out = super(NewBinaryLayer, self).forward(x)
        if(self.verbose):
            print("computing wx + b ")
            print(out)

        self.weight.data = backup_weight
        #          #### CHECK GRADIENTS IN BACKWARD FLOW
        #         gradients = torch.FloatTensor([[1.0,0.0]])
        #         out.backward(gradients)

        #         print(x.grad)
        #         print out.grad, self.weight.grad, self.bias.grad, x.grad
        return F.log_softmax(out)