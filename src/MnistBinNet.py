from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from custom import *
import copy
from MnistNet import *
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 10000)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


eps_const = 1e-5



class Net(nn.Module):
    def __init__(self, binary, stochastic=False):
        super(Net, self).__init__()
        self.binary = binary
        if self.binary:
            self.conv1 = NewBinaryConv2D(1, 10, stochastic=stochastic, kernel_size=5)
            self.conv2 = NewBinaryConv2D(10, 20,  stochastic=stochastic, kernel_size=5)
            self.fc1 = NewBinaryLayer(20*20*20, 50,  stochastic=stochastic)
            self.fc2 = NewBinaryLayer(50, 10, stochastic=stochastic)
            # self.fc2 = NewBinaryLayer(24*24*10, 10, stochastic=stochastic)
        else:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(20*20*20, 50)
            self.fc2 = nn.Linear(50, 10)
            # self.fc2 = nn.Linear(24*24*10, 10, stochastic=stochastic)
        self.bn1 = nn.BatchNorm2d(10,eps=eps_const)
        self.bn2 = nn.BatchNorm2d(20,eps=eps_const)


    def forward(self, x):
        # # RELU SEEMS TO PERFORM FAIRLY POORLY IN THIS NETWORK FOR BINARIZATION :-(
        # x = F.tanh(self.bn1(self.conv1(x)))
        # x = x.view(-1, 24*24*10)
        # x = self.fc2(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 20*20*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class SimpleNet(nn.Module):
    def __init__(self, binary, stochastic=False):
        super(SimpleNet, self).__init__()
        self.binary = binary
        if self.binary:
            self.conv1 = NewBinaryConv2D(1, 10, stochastic=stochastic, kernel_size=5)
            self.fc1 = NewBinaryLayer(24*24*10, 10, stochastic=stochastic)
        else:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.fc1 = nn.Linear(24*24*10, 10)
        self.bn1 = nn.BatchNorm2d(10,eps=eps_const)            

    def forward(self, x):
        x = F.tanh(self.bn1(self.conv1(x)))
        x = x.view(-1, 24*24*10)
        x = self.fc1(x)
        return F.log_softmax(x)



class LinearNet(nn.Module):
    def __init__(self, binary, stochastic=False):
        super(LinearNet, self).__init__()
        self.binary = binary
        if self.binary:
            self.fc1 = NewBinaryLayer(28*28*1, 10, stochastic=stochastic)
        else:
            self.fc1 = nn.Linear(28*28*1, 10)            

    def forward(self, x):
        x = x.view(-1, 28*28*1)

        x = self.fc1(x)
        # x = F.tanh(self.fc1(x))
        return F.log_softmax(x)

class ThreeLayerNet(nn.Module):
    def __init__(self, binary, stochastic=False):
        super(ThreeLayerNet, self).__init__()
        self.binary = binary
        if self.binary:
            self.fc1 = NewBinaryLayer(28*28*1, 2048, stochastic=stochastic)
            self.bn1 = nn.BatchNorm1d(2048,eps=eps_const)
            self.fc2 = NewBinaryLayer(2048, 2048, stochastic=stochastic)
            self.bn2 = nn.BatchNorm1d(2048,eps=eps_const)
            self.fc3 = NewBinaryLayer(2048, 2048, stochastic=stochastic)
            self.bn3 = nn.BatchNorm1d(2048,eps=eps_const)
            self.fc4 = NewBinaryLayer(2048, 10, stochastic=stochastic)
        else:
            self.fc1 = nn.Linear(28*28*1, 2048)  
            self.bn1 = nn.BatchNorm1d(2048,eps=eps_const)
            self.fc2 = nn.Linear(2048, 2048)
            self.bn2 = nn.BatchNorm1d(2048,eps=eps_const)
            self.fc3 = nn.Linear(2048, 2048)
            self.bn3 = nn.BatchNorm1d(2048,eps=eps_const)
            self.fc4 = nn.Linear(2048, 10)          

    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return F.log_softmax(x)
#ST, DET
# model = SimpleNet(True, False)

model = MNISTNET(True,True)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.hinge_embedding_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.hinge_embedding_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def trainacc():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.hinge_embedding_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(train_loader.dataset)
    print('\Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
    trainacc()
