import torch
import torch.nn as nn
import torch.nn.init as init


class simpleNet(torch.nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.layer1 = torch.nn.Linear(2, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 3)
        self.activation = torch.nn.Tanh()
        # self.activation = torch.nn.ReLU()
        # Apply weight initialization to each layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight) # Xavier initialization, good for tanh
                # init.kaiming_normal_(m.weight) # He initialization, good for ReLU

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x
    