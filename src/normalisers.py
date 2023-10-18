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
        if y is not None:
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
        else:
            y = None
        return X, outputs, y
    def denorm_u_r(self, u_r):
        std_y = self.std_y[0]
        mean_y = self.mean_y[0]
        return u_r * std_y + mean_y
    def denorm_r(self, r):
        std = self.std[0]
        mean = self.mean[0]
        return r * std + mean
    def dUrtdUr(self):
        std_y = self.std_y[0]
        return 1/ std_y
    def dUztdUz(self):
        std_y = self.std_y[1]
        return 1/ std_y
    def drdrt(self):
        std = self.std[0]
        return std
    def dzdzt(self):
        std = self.std[1]
        return std
    def denorm_u_z(self, u_z):
        std_y = self.std_y[1]
        mean_y = self.mean_y[1]
        return u_z * std_y + mean_y
    def dPtdP(self):
        std_y = self.std_y[2]
        return 1/ std_y
    
class min_max_normaliser():
    def __init__(self, X, y,constants):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.min_y = np.min(y, axis=0)
        self.max_y = np.max(y, axis=0)
        self.physical = False
    def normalise(self, X,y):
        X = (X - self.min) / (self.max - self.min)
        if y is not None:
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
    def denorm_u_r(self, u_r):
        max_ = self.max_y[0]
        min_ = self.min_y[0]
        return u_r * (max_ - min_) + min_
    def denorm_r(self, r):
        max_ = self.max[0]
        min_ = self.min[0]
        return r * (max_ - min_) + min_
    def dUrtdUr(self):
        max_ = self.max_y[0]
        min_ = self.min_y[0]
        return 1/ (max_ - min_)
    def dUztdUz(self):
        max_ = self.max_y[1]
        min_ = self.min_y[1]
        return 1/ (max_ - min_)
    def drdrt(self):
        max_ = self.max[0]
        min_ = self.min[0]
        return (max_ - min_)
    def dzdzt(self):
        max_ = self.max[1]
        min_ = self.min[1]
        return (max_ - min_)
    def denorm_u_z(self, u_z):
        max_ = self.max_y[1]
        min_ = self.min_y[1]
        return u_z * (max_ - min_) + min_
    def dPtdP(self):
        max_ = self.max_y[2]
        min_ = self.min_y[2]
        return 1/ (max_ - min_)
        
class physics_normaliser():
    def __init__(self,X,y,constants):
        self.physical = True
        self.constants = constants
    def normalise(self, X,y):
        X = X / self.constants["D"]
        rho = self.constants["rho"]
        U_inf = self.constants["U_inf"]
        if y is not None:
            uv = y[:,0:2]
            p = y[:,2].reshape(-1,1)
            uv = uv / U_inf
            p = p / (rho * U_inf**2)
            return X, np.concatenate((uv,p),axis=1)
        else:
            return X, None
    
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
    