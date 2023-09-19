import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn


class two_dim_dataset(Dataset):
    def __init__(self, X, y, min_x, max_x, min_y, max_y):
        self.X = X
        self.y = y
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.X = (self.X - self.min_x)/(self.max_x - self.min_x)
        self.y = (self.y - self.min_y)/(self.max_y - self.min_y)
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()
        self.len = self.X.shape[0]
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def load_data(csv_path,test_size=0.2, random_state=42, drop_hub= True):
    df = pd.read_csv(csv_path)
    # Drop the specified rows
    if drop_hub:
        df = df.drop(df[(np.sqrt(df['x']**2 + df['y']**2) <= 1*178.3)].index)
    X = df[['x', 'y']].values
    y = df[['U', 'V', 'P']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    min_x = X_train.min(axis=0)   
    max_x = X_train.max(axis=0)
    min_y = y_train.min(axis=0)
    max_y = y_train.max(axis=0)
    return X_train, X_test, y_train, y_test, min_x, max_x, min_y, max_y

def plot(X, outputs, y, fig_prefix=""):
    fig = plt.figure(figsize=(15, 5))
    # Plot for U
    ax1 = fig.add_subplot(131)
    sc1 = ax1.scatter(X[:, 0], X[:, 1], c=outputs[:, 0], cmap='viridis')
    ax1.set_title('U')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label('U values')

    # Plot for V
    ax2 = fig.add_subplot(132)
    sc2 = ax2.scatter(X[:, 0], X[:, 1], c=outputs[:, 1], cmap='plasma')
    ax2.set_title('V')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('V values')

    # Plot for P
    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(X[:, 0], X[:, 1], c=outputs[:, 2], cmap='inferno')
    ax3.set_title('P')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cbar3 = fig.colorbar(sc3, ax=ax3)
    cbar3.set_label('P values')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"Figures/{fig_prefix}_2d_cart_NN.pdf")
    plt.show()

    # plot the actual values
    fig = plt.figure(figsize=(15, 5))
    # Plot for U
    ax1 = fig.add_subplot(131)
    sc1 = ax1.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='viridis')
    ax1.set_title('U')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label('U values')

    # Plot for V
    ax2 = fig.add_subplot(132)  
    sc2 = ax2.scatter(X[:, 0], X[:, 1], c=y[:, 1], cmap='plasma')
    ax2.set_title('V')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('V values')

    # Plot for P
    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(X[:, 0], X[:, 1], c=y[:, 2], cmap='inferno')
    ax3.set_title('P')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cbar3 = fig.colorbar(sc3, ax=ax3)
    cbar3.set_label('P values')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"Figures/{fig_prefix}_2d_cart_actual.pdf")
    plt.show()


    # plot error
    error = np.abs(y - outputs)
    fig = plt.figure(figsize=(15, 5))
    # Plot for U
    ax1 = fig.add_subplot(131)
    sc1 = ax1.scatter(X[:, 0], X[:, 1], c=error[:, 0], cmap='viridis')
    ax1.set_title('U')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label('U values')

    # Plot for V
    ax2 = fig.add_subplot(132)
    sc2 = ax2.scatter(X[:, 0], X[:, 1], c=error[:, 1], cmap='plasma')
    ax2.set_title('V')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('V values')

    # Plot for P
    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(X[:, 0], X[:, 1], c=error[:, 2], cmap='inferno')
    ax3.set_title('P')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cbar3 = fig.colorbar(sc3, ax=ax3)
    cbar3.set_label('P values')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"Figures/{fig_prefix}_2d_cart_error.pdf")
    plt.show()


def plot_heatmaps(X, outputs, y, fig_prefix=""):

    def plot_single_heatmap(x, y, values, ax, cmap, cbar_label):
        df = pd.DataFrame.from_dict({'x': x, 'y': y, 'values': values})
        pivoted = df.pivot("y", "x", "values")
        sns.heatmap(pivoted, ax=ax, cmap=cmap, cbar_kws={'label': cbar_label})
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    def plot_all(title, data, file_suffix):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot U
        plot_single_heatmap(X[:, 0], X[:, 1], data[:, 0], ax1, 'jet', 'U values')
        ax1.set_title('U')
        
        # Plot V
        plot_single_heatmap(X[:, 0], X[:, 1], data[:, 1], ax2, 'jet', 'V values')
        ax2.set_title('V')
        
        # Plot P
        plot_single_heatmap(X[:, 0], X[:, 1], data[:, 2], ax3, 'jet', 'P values')
        ax3.set_title('P')

        # Adjust layout and save
        plt.tight_layout()
        plt.suptitle(title, y=1.05)
        plt.savefig(f"Figures/{fig_prefix}_2d_cart_{file_suffix}.pdf")
        plt.show()

    # Plot everything
    plot_all("Predicted", outputs, "NN")
    plot_all("Actual", y, "actual")
    plot_all("Error", np.abs(y - outputs), "error")


def physics_informed_loss(xy, net):
    # AUTOGRAD VERSION.

    xy.requires_grad = True
    uv = net(xy)

    # Compute the Jacobian matrix of uv with respect to xy.
    J = torch.autograd.functional.jacobian(net, xy)
    J_u = J[:, 0, :, :]  # Jacobian of u with respect to xy
    J_v = J[:, 1, :, :]  # Jacobian of v with respect to xy

    # Compute the derivatives of u and v with respect to x and y, respectively.
    dux_dx = torch.sum(J_u * torch.unsqueeze(torch.ones_like(xy[:,0]), 1), dim=1) # ! These two functions sums columns i.e. this only works because gradients between observations are all zero in this case (only non-zero values in the diagonal). I think that is always the case with PINN's but I am not sure.
    duy_dy = torch.sum(J_v * torch.unsqueeze(torch.ones_like(xy[:,0]), 1), dim=1)
    
    # # Compute the mass conservation equation.
    mass_conservation = dux_dx[:,0] + duy_dy[:,1]

    loss_f = nn.MSELoss()
    loss = loss_f(mass_conservation, torch.zeros_like(mass_conservation))

    return loss