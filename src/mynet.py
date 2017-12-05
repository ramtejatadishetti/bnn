
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
from binaryutility import *

class ClippedOptimizer(optim.SGD):
    def __init__(self, params, H, lr,binary, momentum=0, dampening=0,weight_decay=0, nesterov=False):
        super(ClippedOptimizer, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.H = H
        self.binary = binary


    def step(self, closure=None):
        super(ClippedOptimizer, self).step(closure)
        
        if self.binary:
            #print("binary optim")

            for wt in self.param_groups:
                print(wt)
                for p in wt['params']:
                    if p.grad is None:
                        continue
                    p.data = torch.clamp(p.data, -self.H, self.H)


EPOCHS = 1

class MyNet(nn.Module):
    def __init__(self, binary):
        super(MyNet, self).__init__()

        self.binary = binary

        self.binary_conv1 = BinaryConvLayer(3,6,5,1, self.binary)
        self.binary_conv2 = BinaryConvLayer(6, 16, 5, 1, self.binary)

        self.binary_linear1 = BinaryLayer(16*5*5, 120, 1, self.binary)
        self.binary_linear2 = BinaryLayer(120, 84, 1, self.binary)
        self.binary_linear3 = BinaryLayer(84, 10, 1, False)
    
    def forward(self, x):
        out = F.relu(self.binary_conv1(x))
        out = F.max_pool2d(out,2)
        out = F.relu(self.binary_conv2(out))
        out = F.max_pool2d(out,2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.binary_linear1(out))
        out = F.relu(self.binary_linear2(out))
        out = self.binary_linear3(out)
    
        return out

if __name__ == "__main__":

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = MyNet(True)
    #print(net)
    #print(net.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = ClippedOptimizer(net.parameters(), 1, 0.001, True)

    '''
    variables = net.state_dict()
    for key in variables.keys():
        print(key)
    '''
    
    for epoch in range(1):
        print('\nEpoch: %d ' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        batch_idx = 0
        inputs = None
        targets = None

        for batch_idx , (inputs, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            variables = net.state_dict()
            #for key in variables.keys():
            #    print(key)
            #break    
            
            optimizer.step()
            break


    

    
