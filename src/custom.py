from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
import random
#hard sigmoid functiom
def hard_sigmoid(x):
    return torch.clamp((x+1.)/2., 0, 1)

def binarize(W, stochastic=False):
    x = copy.deepcopy(W.data)
    y = torch.clamp(x, -1, 1)
    x = hard_sigmoid(y)
    m = copy.deepcopy(x)
    if(stochastic):
        x = torch.bernoulli(m)
    else:
        x = torch.round(x)
    x[x==0] = -1
    # if(random.random()<=0.01):
        # print(x) 
    return x,y

def quantized_binarize(W, H):
    x = copy.deepcopy(W.data)
    y = torch.clamp(x, -H, H)

    x = torch.round(x)

    return x,y


class NewBinaryLayer(nn.Linear):
    #initialize the Binary Layer where weights are binarized
    def __init__(self, input_dim, output_dim, stochastic=False, quantization=1,verbose=False):
        self.verbose = verbose
        self.stochastic = stochastic
        self.quantization = quantization
        super(NewBinaryLayer, self).__init__(input_dim, output_dim)
        
        
    def forward(self, x):
        if(self.verbose):
            print("Weights,bias in forward prop before binarization")
            print(self.weight.data)
            print(self.bias.data)
        
        if self.quantization == 1:
            self.new_weight,clipped_wt_data = binarize(self.weight, self.stochastic)
        
        else:
            self.new_weight,clipped_wt_data = quantized_binarize(self.weight, self.quantization)

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
        return out



class NewBinaryConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stochastic=False, quantization=1, **kwargs):
        self.stochastic = stochastic
        self.quantization = quantization
        super(NewBinaryConv2D, self).__init__(in_channels, out_channels, **kwargs)
    
    def forward(self, x):

        if self.quantization == 1:
            self.new_weight, clipped_wt_data = binarize(self.weight, self.stochastic)
        
        else:
            self.new_weight,clipped_wt_data = quantized_binarize(self.weight, self.quantization)

        # replace the weights with clipped weights
        # this part should be done in parameter update
        # but make it here still does not corrupt the data
        # in forward prop or backward prop 
        self.weight.data = clipped_wt_data

        # store the old weights so that they could be restored later
        backup_weight = self.weight.data

        # replace the binary weights into actual weights
        self.weight.data = self.new_weight


        # compute layer operation
        out = super(NewBinaryConv2D, self).forward(x)

        # restore old weights
        self.weight.data = backup_weight

        return out


class NewRound(torch.autograd.Function):
    #initialize the Binary Activation Function after Tanh
    def __init__(self):
        super(NewRound, self).__init__()
        
    def forward(self, input, stochastic=True):
        out = torch.round(input)
        return out
    
    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()
        return grad_input


class Binaryactivation(torch.autograd.Function):
    #initialize the Binary Activation Function after Tanh
    def __init__(self):
        super(Binaryactivation, self).__init__()
        
    def forward(self, input, stochastic=True):
        x = (2*hard_sigmoid(input) - 1)
        out = NewRound()(x)
        return out


class BinarytanH(torch.autograd.Function):
    #initialize the Binary Activation Function after Tanh
    def __init__(self):
        super(BinarytanH, self).__init__()
        
    def forward(self, x):
        out = Binaryactivation().forward(x)
        return out

class BintanH(torch.nn.Module):
    #initialize the Binary Activation Function after Tanh
    def __init__(self):
        super(BintanH, self).__init__()
        
    def forward(self, x):
        out = Binaryactivation().forward(x)
        return out

