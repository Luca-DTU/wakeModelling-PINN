import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn


class normalised_dataset(Dataset):
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
    
class dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()
        self.len = self.X.shape[0]
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    

def load_data(csv_path,test_size=0.2, random_state=42, drop_hub= True, D = 0):
    df = pd.read_csv(csv_path)
    # Drop the specified rows
    if drop_hub:
        df = df.drop(df[(np.sqrt(df['r']**2 + df['z_cyl']**2) <= D)].index)
    X = df[['r', 'z_cyl']].values
    y = df[['Ux', 'Ur', 'P']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    min_x = X_train.min(axis=0)   
    max_x = X_train.max(axis=0)
    min_y = y_train.min(axis=0)
    max_y = y_train.max(axis=0)
    return df ,X_train, X_test, y_train, y_test, min_x, max_x, min_y, max_y

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

    def plot_single_heatmap(r, z, values, ax, cmap, cbar_label):
        df = pd.DataFrame.from_dict({'r': r, 'z': z, 'values': values})
        pivoted = df.pivot("r", "z", "values")
        sns.heatmap(pivoted, ax=ax, cmap=cmap, cbar_kws={'label': cbar_label})
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('z')
        ax.set_ylabel('r')

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


def physics_informed_loss(rz, net, constants, remove_dimensionality = False):
    # Given values
    rho = constants["rho"]
    mu = constants["mu"]
    mu_t = constants["mu_t"]
    nu = mu / rho  # kinematic viscosity
    nu_total = nu + mu_t / rho  # total kinematic viscosity (laminar + turbulent)
    r = rz[:, 0]
    # set up input
    rz.requires_grad = True
    uvp = net(rz)
    u_r = uvp[:, 0]
    u_z = uvp[:, 1]
    p = uvp[:, 2]
    # Calculate the gradients
    du_r = torch.autograd.grad(u_r, rz, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
    du_rdr,du_rdz = du_r[:, 0], du_r[:, 1]
    du_z = torch.autograd.grad(u_z, rz, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    du_zdr,du_zdz = du_z[:, 0], du_z[:, 1]
    dp = torch.autograd.grad(p, rz, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dpdr,dpdz = dp[:, 0], dp[:, 1]
    # second derivatives
    d2u_rdr2 = torch.autograd.grad(du_rdr, rz, grad_outputs=torch.ones_like(du_rdr), create_graph=True)[0][:, 0]
    d2u_rdz2 = torch.autograd.grad(du_rdz, rz, grad_outputs=torch.ones_like(du_rdz), create_graph=True)[0][:, 1]
    d2u_zdr2 = torch.autograd.grad(du_zdr, rz, grad_outputs=torch.ones_like(du_zdr), create_graph=True)[0][:, 0]
    d2u_zdz2 = torch.autograd.grad(du_zdz, rz, grad_outputs=torch.ones_like(du_zdz), create_graph=True)[0][:, 1]
    # mass conservation 1/r d(ru_r)/dr + du_z/dz = 0 => du_r/dr + u_r/r + du_z/dz +  = 0
    mass_conservation = du_rdr + u_r/r + du_zdz
    # r-momentum conservation u_r du_r/dr + u_z du_r/dz + 1/rho dp/dr - nu_total (1/r du_r/dr + d2u_r/dr2 + d2u_r/dz2 - u_r/r^2) = 0
    # z-momentum conservation u_r du_z/dr + u_z du_z/dz + 1/rho dp/dz - nu_total (1/r du_z/dr + d2u_z/dr2 + d2u_z/dz2) = 0
    # r-momentum
    r_momentum = u_r*du_rdr + u_z*du_rdz + 1/rho*dpdr - nu_total*(1/r*du_rdr + d2u_rdr2 + d2u_rdz2 - u_r/r**2)
    # z-momentum
    z_momentum = u_r*du_zdr + u_z*du_zdz + 1/rho*dpdz - nu_total*(1/r*du_zdr + d2u_zdr2 + d2u_zdz2)
    # return the loss
    loss = torch.mean(mass_conservation**2 + r_momentum**2 + z_momentum**2)
    return loss

def plot_losses(losses, fig_prefix):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(losses["collocation"])
    ax1.set_title("Collocation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax2 = fig.add_subplot(122)
    ax2.plot(losses["physics"])
    ax2.set_title("Physics loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"Figures/{fig_prefix}_losses.pdf")
    plt.show()