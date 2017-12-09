from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
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


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1440*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.tanh((self.conv1(x)))
        x = x.view(-1, 1440*4)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


# class NonBinLinearNet(nn.Module):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self.fc1 = nn.Linear(28*28*1, 10)

#     def forward(self, x):
#         x = x.view(-1, 28*28*1)
#         x = self.fc1(x)
#         # x = F.tanh(self.fc1(x))
#         return F.log_softmax(x)

class LinearNet(nn.Module):
    def __init__(self, binary):
        super(LinearNet, self).__init__()
        self.binary = binary
        if self.binary:
            self.fc1 = NewBinaryLayer(28*28*1, 10)
        else:
            self.fc1 = nn.Linear(28*28*1, 10)            

    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = self.fc1(x)
        # x = F.tanh(self.fc1(x))
        return F.log_softmax(x)

def binarize(W, stochastic=False):
    x = copy.deepcopy(W.data)
    x[x>0] = 1
    x[x<=0] = -1
    return x

class NewBinaryLayer(nn.Linear):
    #initialize the Binary Layer where weights are binarized
    def __init__(self, input_dim, output_dim):
        super(NewBinaryLayer, self).__init__(input_dim, output_dim)
        
    def forward(self, x):
        self.new_weight = binarize(self.weight)
        # print self.new_weight.grad_fn
        backup_weight = self.weight.data
        self.weight.data = backup_weight
        out = super(NewBinaryLayer, self).forward(x)
        self.weight.data = backup_weight
        return out

model = LinearNet()
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
        loss = F.nll_loss(output, target)
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
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
