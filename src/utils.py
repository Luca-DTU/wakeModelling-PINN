import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn

    
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
    y = df[['Ur','Ux', 'P']].values #Ux is actually Uz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return df ,X_train, X_test, y_train, y_test


def physics_informed_loss(rz, net, constants):
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

def print_graph(g, indent=''):
    """Prints the computation graph"""
    if g is None:
        return
    print(indent, g)
    for next_g in g.next_functions:
        if next_g[0] is not None:
            print_graph(next_g[0], indent + '  ')

def non_dimensionalized_physics_informed_loss(rz, net, constants,Normaliser):
    if isinstance(Normaliser,str):
        physical_normaliser = Normaliser.physical
    elif isinstance(Normaliser,list):
        physical_normaliser = any([Normaliser_.physical for Normaliser_ in Normaliser])
    # Given values and characteristic scales
    rho = constants["rho"]
    mu = constants["mu"]
    mu_t = constants["mu_t"]
    
    U_inf = constants["U_inf"]  # Characteristic velocity
    D = constants["D"]        # Characteristic length
    P = rho * U_inf*U_inf    # Characteristic pressure
    
    # Non-dimensional parameters
    nu = mu / (rho * U_inf * D)  # Non-dimensional kinematic viscosity
    nu_t = mu_t / (rho * U_inf * D)  # Non-dimensional turbulent kinematic viscosity (eddy viscosity)
    
    # Non-dimensional coordinates and variables
    if physical_normaliser:
        rz = rz #/ D  # Non-dimensional coordinates
    else:
        rz = rz / D
    r = rz[:, 0]
    
    # set up input
    rz.requires_grad = True
    uvp = net(rz)  # Non-dimensional velocities and pressure
    ###
    # note: this approach assumes that we do physical normalisation first and then the numerical ones (min-max or z-score)
    # remember to raise an error if the order is wrong at the beginning of main()
    # if isinstance(Normaliser,list):
    #     for Normaliser_ in Normaliser[::-1]: 
    #         if not Normaliser_.physical: 
    #             rz, uvp, _ = Normaliser_.denormalise(rz, uvp, None)
    ###
    # print_graph(uvp.grad_fn)
    # print_graph(rz.grad_fn)
    u_r = uvp[:, 0]
    u_z = uvp[:, 1]
    if physical_normaliser:
        p = uvp[:, 2]
    else:
        p = uvp[:, 2] / P  # Non-dimensional pressure
    # Calculate the gradients
    du_r = torch.autograd.grad(u_r, rz, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
    du_r_dr, du_r_dz = du_r[:, 0], du_r[:, 1]
    du_z = torch.autograd.grad(u_z, rz, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    du_z_dr, du_z_dz = du_z[:, 0], du_z[:, 1]
    dp = torch.autograd.grad(p, rz, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dr, dp_dz = dp[:, 0], dp[:, 1]
    # second derivatives
    d2u_r_dr2 = torch.autograd.grad(du_r_dr, rz, grad_outputs=torch.ones_like(du_r_dr), create_graph=True)[0][:, 0]
    d2u_r_dz2 = torch.autograd.grad(du_r_dz, rz, grad_outputs=torch.ones_like(du_r_dz), create_graph=True)[0][:, 1]
    d2u_z_dr2 = torch.autograd.grad(du_z_dr, rz, grad_outputs=torch.ones_like(du_z_dr), create_graph=True)[0][:, 0]
    d2u_z_dz2 = torch.autograd.grad(du_z_dz, rz, grad_outputs=torch.ones_like(du_z_dz), create_graph=True)[0][:, 1]
    # Mass conservation equation in non-dimensional form
    # mass_conservation = du_r_dr + u_r / r + du_z_dz
    n = Normaliser[-1]
    t1 = n.denorm_u_r(u_r)/n.denorm_r(r) # u_r / r
    t2 = du_r_dr / (n.dUrtdUr() * n.drdrt()) # du_r_dr
    t3 = du_z_dz / (n.dUztdUz() * n.dzdzt()) # du_z_dz
    mass_conservation = t1 + t2 + t3
    # # r-momentum and z-momentum equations in non-dimensional form
    # r_momentum = u_r * du_r_dr + u_z * du_r_dz + dp_dr - \
    #              (nu + nu_t) * (1 / r * du_r_dr + d2u_r_dr2 + d2u_r_dz2 - u_r / r**2)
    # second derivatives:
    d2u_r_dz2_ = d2u_r_dz2/ (n.dUrtdUr() * n.dzdzt()**2)
    d2u_r_dr2_ = d2u_r_dr2/ (n.dUrtdUr() * n.drdrt()**2)
    d2u_z_dz2_ = d2u_z_dz2/ (n.dUztdUz() * n.dzdzt()**2)
    d2u_z_dr2_ = d2u_z_dr2/ (n.dUztdUz() * n.drdrt()**2)

    r_momentum = n.denorm_u_r(u_r) * du_r_dr / (n.dUrtdUr() * n.drdrt()) + \
                n.denorm_u_z(u_z) * du_r_dz / (n.dUrtdUr() * n.dzdzt()) + \
                dp_dr / (n.dPtdP() * n.drdrt()) - (nu + nu_t) * (1 / n.denorm_r(r) * du_r_dr / (n.dUrtdUr() * n.drdrt()) + d2u_r_dr2_ + d2u_r_dz2_ - n.denorm_u_r(u_r) / n.denorm_r(r)**2)
    
    # z_momentum = u_r * du_z_dr + u_z * du_z_dz + dp_dz - \
    #              (nu + nu_t) * (1 / r * du_z_dr + d2u_z_dr2_ + d2u_z_dz2_)
    z_momentum = n.denorm_u_r(u_r) * du_z_dr / (n.dUztdUz() * n.drdrt()) + \
                n.denorm_u_z(u_z) * du_z_dz / (n.dUztdUz() * n.dzdzt()) + \
                dp_dz / (n.dPtdP() * n.dzdzt()) - (nu + nu_t) * (1 / n.denorm_r(r) * du_z_dr / (n.dUztdUz() * n.drdrt()) + d2u_z_dr2_ + d2u_z_dz2_)
    
    # # Return the non-dimensionalized loss
    loss = torch.mean(mass_conservation**2 + r_momentum**2 + z_momentum**2)
    # return loss
    # loss = torch.mean(mass_conservation**2)
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
        plot_single_heatmap(X[:, 0], X[:, 1], data[:, 0], ax1, 'jet', 'U_r values')
        ax1.set_title('U_r')
        
        # Plot V
        plot_single_heatmap(X[:, 0], X[:, 1], data[:, 1], ax2, 'jet', 'U_z values')
        ax2.set_title('U_z')
        
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
