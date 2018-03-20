class Net(nn.Module):
    def __init__(self, binary, stochastic=False):
        super(Net, self).__init__()
        self.binary = binary
        if self.binary:
            self.conv1 = NewBinaryConv2D(1, 10, stochastic=stochastic, kernel_size=5)
            self.conv2 = NewBinaryConv2D(10, 20,  stochastic=stochastic, kernel_size=5)
            self.fc1 = NewBinaryLayer(320, 50,  stochastic=stochastic)
            self.fc2 = NewBinaryLayer(50, 10, stochastic=stochastic)
        else:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        # RELU SEEMS TO PERFORM FAIRLY POORLY IN THIS NETWORK FOR BINARIZATION :-(
        x = F.tanh(F.max_pool2d(self.conv1(x), 2))
        x = F.tanh(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)