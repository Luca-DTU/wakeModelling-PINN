import torch
import torch.nn as nn
import torch.nn.init as init

class residualNetBigData(nn.Module):
    def __init__(self):
        super(residualNetBigData, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 3)
        self.activation = torch.nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x[:,:4]
        out = self.activation(self.fc1(x))
        identity = out  # save the output for the skip connection
        out = self.activation(self.fc2(out)) 
        out = self.activation(self.fc3(out)) + identity
        out = self.fc4(out)
        return out

class simpleNetBigData(torch.nn.Module):
    def __init__(self):
        super(simpleNetBigData, self).__init__()
        self.layer1 = torch.nn.Linear(4, 64)
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
        x = x[:,:4]
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x

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
class broaderNet(torch.nn.Module):
    def __init__(self):
        super(broaderNet, self).__init__()
        self.layer1 = torch.nn.Linear(2, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, 3)
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

class deeperNet(nn.Module):
    def __init__(self):
        super(deeperNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 3)
        self.activation = torch.nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class residualNet(nn.Module):
    def __init__(self):
        super(residualNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 3)
        self.activation = torch.nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        out = self.activation(self.fc1(x))
        identity = out  # save the output for the skip connection
        out = self.activation(self.fc2(out)) 
        out = self.activation(self.fc3(out)) + identity
        out = self.fc4(out)
        return out

