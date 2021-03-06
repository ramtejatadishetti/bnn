from custom import *

class MNISTNET(nn.Module):
    def __init__(self, bin = False, stochastic = False):
        super(MNISTNET, self).__init__()
        self.bin = bin
        self.bin_act = False

        self.stochastic = stochastic
        self.features1 = self._make_layers()
        self.features2 = self._make_layers()
        self.features3 = self._make_layers()
        if(not self.bin):
            self.classifier = nn.Linear(4096, 10)
            self.fc1 = nn.Linear(28*28,4096)
        else:
            self.classifier = NewBinaryLayer(4096, 10, stochastic=self.stochastic)
            self.fc1 = NewBinaryLayer(28*28,4096, stochastic=self.stochastic)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if(self.bin_act):
            x = BinarytanH().forward(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
        out = self.features1(x)
        if(self.bin_act):
            out = BinarytanH().forward(out)
        else:
            out = F.relu(out)
        # out = self.features2(out)
        # if(self.bin_act):
        #     out = BinarytanH().forward(out)
        # else:
        #     out = F.relu(out)
        # out = self.features3(out)
        # if(self.bin_act):
        #     out = BinarytanH().forward(out)
        # else:
        #     out = F.relu(out)
        out = self.classifier(out)
        out = self.bn(out)
        return (out)

    def _make_layers(self):
        layers = []
        num_units = 4096
        if(self.bin):
            layers += [NewBinaryLayer(num_units, num_units, self.stochastic),nn.BatchNorm1d(num_units)]
        else:
            layers += [nn.Linear(num_units, num_units, self.stochastic),nn.BatchNorm1d(num_units)]
        return nn.Sequential(*layers)
