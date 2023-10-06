import numpy as np
import torch

class Z_normaliser():
    def __init__(self, X, y,constants):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.mean_y = np.mean(y, axis=0)
        self.std_y = np.std(y, axis=0)
        self.physical = False
    def normalise(self, X,y):
        X = (X - self.mean) / self.std
        y = (y - self.mean_y) / self.std_y
        return X,y
    def denormalise(self, X,outputs,y):
        if isinstance(X,torch.Tensor):
            std = torch.from_numpy(self.std).float()
            mean = torch.from_numpy(self.mean).float()
            std_y = torch.from_numpy(self.std_y).float()
            mean_y = torch.from_numpy(self.mean_y).float()
        else:
            std = self.std
            mean = self.mean
            std_y = self.std_y
            mean_y = self.mean_y
        X = X * std + mean
        outputs = outputs * std_y + mean_y
        if y is not None:
            y = y * std_y + mean_y
        return X, outputs, y
class min_max_normaliser():
    def __init__(self, X, y,constants):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.min_y = np.min(y, axis=0)
        self.max_y = np.max(y, axis=0)
        self.physical = False
    def normalise(self, X,y):
        X = (X - self.min) / (self.max - self.min)
        y = (y - self.min_y) / (self.max_y - self.min_y)
        return X,y
    def denormalise(self, X,outputs,y):
        if isinstance(X,torch.Tensor):
            max_ = torch.from_numpy(self.max).float().to(X.device)
            min_ = torch.from_numpy(self.min).float().to(X.device)
            max_y = torch.from_numpy(self.max_y).float().to(outputs.device)
            min_y = torch.from_numpy(self.min_y).float().to(outputs.device)
        else:
            max_ = self.max
            min_ = self.min
            max_y = self.max_y
            min_y = self.min_y
        X_ = X * (max_ - min_) + min_
        outputs_ = outputs * (max_y - min_y) + min_y
        if y is not None:
            y_ = y * (max_y - min_y) + min_y
        else:
            y_ = None
        return X_, outputs_, y_
        
class physics_normaliser():
    def __init__(self,X,y,constants):
        self.physical = True
        self.constants = constants
    def normalise(self, X,y):
        X = X / self.constants["D"]
        rho = self.constants["rho"]
        U_inf = self.constants["U_inf"]
        uv = y[:,0:2]
        p = y[:,2].reshape(-1,1)
        uv = uv / U_inf
        p = p / (rho * U_inf**2)
        return X, np.concatenate((uv,p),axis=1)
    
    def denormalise(self, X,outputs,y):
        # X
        X = X * self.constants["D"]
        # y
        rho = self.constants["rho"]
        U_inf = self.constants["U_inf"]
        uv = y[:,0:2]
        p = y[:,2].reshape(-1,1)
        uv = uv * U_inf
        p = p * rho * U_inf**2
        y = np.concatenate((uv,p),axis=1)
        # outputs
        uv = outputs[:,0:2]
        p = outputs[:,2].reshape(-1,1)
        uv = uv * U_inf
        p = p * rho * U_inf**2
        outputs = np.concatenate((uv,p),axis=1)
        return X, outputs, y
    