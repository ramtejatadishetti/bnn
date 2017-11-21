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
                
    
    def forward(self, input_weights):

        if self.binary:
            #print("binary forward bin")
            self.Wb = binarization(self.weight, self.H, self.binary)
            backup_weight = self.weight.data
            self.weight.data = self.Wb.data

        out = super(BinaryLayer, self).forward(input_weights)

        if self.binary:
            self.weight.data = backup_weight

        return out

class BinaryConvLayer(nn.Conv2d):

    #initialize the 
    def __init__(self, input_channels, output_channels, kernel_size, H, binary=True):

        self.H = H
        self.binary = binary
        super(BinaryConvLayer, self).__init__(input_channels, output_channels, kernel_size)
                
    
    def forward(self, input_weights):

        if self.binary:
            #print("binary forward")
            self.Wb = binarization(self.weight, self.H, self.binary)
            backup_weight = self.weight.data
            self.weight.data = self.Wb.data

        out = super(BinaryConvLayer, self).forward(input_weights)
        
        if self.binary:
            self.weight.data = backup_weight

        return out



class MyNetwork(nn.Module):
    def __init__(self, binary):
        super(MyNetwork, self).__init__()

        self.binary = binary

        self.conv1 = BinaryConvLayer(3,6,5,1, self.binary)
        self.conv2 = BinaryConvLayer(6, 16, 5, 1, self.binary)

        self.bc1 = BinaryLayer(16*5*5, 120, 1, self.binary)
        self.bc2 = BinaryLayer(120, 84, 1, self.binary)
        self.bc3 = BinaryLayer(84, 10, 1, self.binary)

    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out,2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out,2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.bc1(out))
        out = F.relu(self.bc2(out))
        out = self.bc3(out)

        return out        





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
                for p in wt['params']:
                    if p.grad is None:
                        continue
                    p.data = torch.clamp(p.data, -self.H, self.H)




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

    #bd_layer = BinaryLayer(3, 4, 1, True)

    net = MyNetwork(False)
    print(net) 
    criterion = nn.CrossEntropyLoss()
    optimizer = ClippedOptimizer(net.parameters(), 1, 0.001, False)

    start_epoch = 1
    best_acc = 0

    for epoch in range(start_epoch, start_epoch+100):
        print('\nEpoch: %d' % epoch)

        net.train()
        train_loss = 0
        correct = 0
        total = 0

        
        batch_idx = 0
        inputs = None
        targets = None

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            #loss = torch.mean(torch.max(0., 1. - outputs.data.numpy()*targets.data.numpy())**2)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        train_acc = 100.*correct/total

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        test_acc = 100.*correct/total 
        print("Epoch, Training accuracy, Test Accuracy", epoch, train_acc, test_acc)
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc
        
        
    '''
    mynet = MyNetwork()
    print(mynet) 

    params = list(mynet.parameters())
    print(len(params))
    
    [print(i) for i in params ]

    
    optimizer = ClippedOptimizer(bd_layer.parameters(),1, 0.01, True)
    optimizer.zero_grad()
    criterion = nn.MSELoss()

    output = bd_layer.forward(inp)

    loss = criterion(output, target)
    loss.backward()

    optimizer.step()
    '''
