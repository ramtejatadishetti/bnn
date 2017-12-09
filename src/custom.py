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