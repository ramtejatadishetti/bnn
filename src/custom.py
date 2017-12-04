from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


#hard sigmoid functiom
def hard_sigmoid(x):
    return torch.clamp((x+1.)/2., 0, 1)

class BinarizeWeights(torch.autograd.Function):
    def __init__(self):
        super(BinarizeWeights, self).__init__()
    
    def forward(self, input, S, stochastic=True):
        self.save_for_backward(S)
        if(stochastic):
            x = hard_sigmoid(input)
            res = torch.bernoulli(x)
            res[res == 0] = -1
        else:
            res = torch.sign()
        return res

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        S, = self.saved_tensors
        grad_input = torch.mm(grad_output, S)
        return grad_input


class BinaryMaxPool2DLayer(nn.MaxPool2d):
    #initialize the Binary Layer where weights are binarized
    def __init__(self, **kwargs):
        super(BinaryMaxPool2DLayer, self).__init__(**kwargs)
        
    def forward(self, x):
        self.new_weight = BinarizeWeights().forward(self.weight,x)
        backup_weight = self.weight.data
        self.weight.data = self.new_weight.data
        out = super(BinaryMaxPool2DLayer, self).forward(x)
        return out

class BinaryAvgPool2DLayer(nn.AvgPool2d):
    #initialize the Binary Layer where weights are binarized
    def __init__(self, **kwargs):
        super(BinaryAvgPool2DLayer, self).__init__(**kwargs)
        
    def forward(self, x):
        self.new_weight = BinarizeWeights().forward(self.weight,x)
        backup_weight = self.weight.data
        self.weight.data = self.new_weight.data
        out = super(BinaryAvgPool2DLayer, self).forward(x)
        return out


class BinaryConv2DLayer(nn.Conv2d):
    #initialize the Binary Layer where weights are binarized
    def __init__(self, input_dim, output_dim, **kwargs):
        super(BinaryConv2DLayer, self).__init__(input_dim, output_dim, **kwargs)
        
    def forward(self, x):
        self.new_weight = BinarizeWeights().forward(self.weight,x)
        backup_weight = self.weight.data
        self.weight.data = self.new_weight.data
        out = super(BinaryConv2DLayer, self).forward(x)
        return out

class BinaryLayer(nn.Linear):
    #initialize the Binary Layer where weights are binarized
    def __init__(self, input_dim, output_dim):
        super(BinaryLayer, self).__init__(input_dim, output_dim)
        
    def forward(self, x):
        self.new_weight = BinarizeWeights().forward(self.weight,x)
        # print self.new_weight.grad_fn
        backup_weight = self.weight.data
        self.weight.data = self.new_weight.data
        out = super(BinaryLayer, self).forward(x)
        return out


class Binaryactivation(torch.autograd.Function):
    #initialize the Binary Activation Function after Tanh
    def __init__(self):
        super(Binaryactivation, self).__init__()
        
    def forward(self, input, stochastic=True):
        self.save_for_backward(input)
        if(stochastic):
            x = hard_sigmoid(input)
            out = torch.bernoulli(x)
            out[out == 0] = -1
        else:
            out = torch.sign(input)
        return out
    
    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) >= 1] = 0
        return grad_input

class BinarytanH(torch.autograd.Function):
    #initialize the Binary Activation Function after Tanh
    def __init__(self):
        super(BinarytanH, self).__init__()
        
    def forward(self, x):
        res = F.tanh(x)
        out = Binaryactivation()(res)
#         print out.grad_fn
        return out