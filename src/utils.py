import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import griddata
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class dataset(Dataset):
    def __init__(self, X, y, batch_size):
        self.batch_size = batch_size
        match X.shape[1]:
            case 2:
                self.X = X
                self.y = y
                self.X = torch.from_numpy(self.X).float()
                self.y = torch.from_numpy(self.y).float()
                self.len = self.X.shape[0]
            case 5:
                self.X = X
                self.y = y
                self.X = torch.from_numpy(self.X).float()
                self.y = torch.from_numpy(self.y).float()
                unique_combinations = np.unique(self.X[:, 2:4], axis=0)
                self.batch_indices = []
                for combination in unique_combinations:
                    indices = np.where((self.X[:, 2] == combination[0]) & (self.X[:, 3] == combination[1]))[0]
                    num_batches = int(np.ceil(len(indices) / self.batch_size))
                    batches = np.array_split(indices, num_batches)
                    self.batch_indices.extend(batches)
                self.len = len(self.batch_indices)
            case _:
                raise NotImplementedError("Only 2D and 5D data supported")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.X.shape[1] == 2:
            return self.X[idx], self.y[idx]
        elif self.X.shape[1] == 5:
            batch_indices = self.batch_indices[idx]
            return self.X[batch_indices], self.y[batch_indices]
    
class test_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        match X.shape[1]:
            case 2:
                self.y = y
                self.X = torch.from_numpy(self.X).float()
                self.y = torch.from_numpy(self.y).float()
                self.len = 1
            case 5:
                self.df = pd.DataFrame(np.concatenate((X, y), axis=1), columns=['col0','col1', 'col2', 'col3', 'col4', 'col5', 'col6','col7'])
                self.groups = self.df.groupby(['col2', 'col3'])
                self.group_dict = {group: torch.from_numpy(data.values).float() for group, data in self.groups}
                self.len = len(self.group_dict) #len(self.groups)
            case _:
                raise NotImplementedError("Only 2D and 5D data supported")
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.X.shape[1] == 2:
            return self.X, self.y
        elif self.X.shape[1] == 5:
            instance = self.group_dict[list(self.groups.groups.keys())[idx]]
            return instance[:,:-3], instance[:,-3:]
    

def sample_phys_points(X_phys,physics_points_size_ratio,p = None):
    num_samples = int(len(X_phys) * physics_points_size_ratio)
    idx = np.random.choice(len(X_phys), num_samples, replace=False, p=p)
    X_phys = X_phys[idx]
    return X_phys

def sample_points(X_mat, n_samples, Normaliser, D=0,grid_size=100):
    out_mat = X_mat.copy()
    for Normaliser_ in Normaliser[::-1]:
        out_mat[:,:2],_,_ = Normaliser_.denormalise(out_mat[:,:2],None,None)
    # Extract x, y, and p values
    x, y, p = out_mat[:,0], out_mat[:,1], out_mat[:,2]
    # Create a grid
    grid_x, grid_y = np.mgrid[min(x):max(x):complex(grid_size), min(y):max(y):complex(grid_size)]
    # Interpolate p values on the grid
    grid_p = griddata((x, y), p, (grid_x, grid_y), method='linear', fill_value=0)
    # Flatten the grid for easy sampling
    points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    probabilities = grid_p.ravel()
    # get the index of the points on the hub
    hub_idx = np.where(np.sqrt(points[:,0]**2 + points[:,1]**2) <= D)[0]
    # set the probabilities of the points on the hub to zero
    probabilities[hub_idx] = 0
    # normalise the probabilities
    probabilities = probabilities / np.sum(probabilities)
    # Sample points based on probabilities
    indices = np.random.choice(len(points), size=n_samples, p=probabilities)
    sampled_points = points[indices]
    # fig,axs = plt.subplots(3,1,figsize=(10,8))
    # axs[0].scatter(X_mat[:,0],X_mat[:,1],c=X_mat[:,2])
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # axs[1].scatter(points[:,0],points[:,1],c=probabilities)
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    # axs[2].scatter(sampled_points[:,0],sampled_points[:,1],s=1)
    # axs[2].set_xticks([])
    # axs[2].set_yticks([])
    # plt.tight_layout()
    # plt.show()
    sampled_points = torch.from_numpy(sampled_points).float().to(device)
    for Normaliser_ in Normaliser:
        sampled_points[:,:2],_ = Normaliser_.normalise(sampled_points[:,:2],None)
    return sampled_points


def load_data(path,test_size=0.2, random_state=42, drop_hub= True, D = 0, shuffle=True, physics_points_size_ratio = 0.1):
    if path.endswith(".nc"):
        ds = xr.open_dataset(path)
        print(ds)
        variables = list(ds.data_vars.keys())
        print(variables)
        df_list = []
        for variable in variables:
            if variable == "muT":
                continue
            df = ds[variable].to_dataframe()
            df_list.append(df)
        df = pd.concat(df_list, axis=1)
        df = df.merge(ds["muT"].to_dataframe(), left_index=True, right_index=True).reset_index()
        if drop_hub:
            df = df.drop(df[(np.sqrt(df['r']**2 + df['z_cyl']**2) <= D)].index)
        df_test = df[df["CT"] == 0.73105] # specific case here, will need to be changed in refactor
        df_train = df[df["CT"].isin([0.63388,0.814])]
        X_train = df_train[['r', 'z_cyl','CT', 'TI_amb',"muT"]].values
        y_train = df_train[['U_r','U_z', 'P']].values
        X_test = df_test[['r', 'z_cyl','CT', 'TI_amb',"muT"]].values
        y_test = df_test[['U_r','U_z', 'P']].values
        single_case = df_test[df_test["TI_amb"] == 0.27]
        X_phys = single_case[['r', 'z_cyl']].sample(frac=physics_points_size_ratio, random_state=random_state).values
        # sample a fraction of training points
        num_samples = int(len(X_train) * (1-test_size))
        idx = np.random.choice(len(X_train), num_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        # Drop the specified rows
        if drop_hub:
            df = df.drop(df[(np.sqrt(df['r']**2 + df['z_cyl']**2) <= D)].index)
        X = df[['r', 'z_cyl']].values
        y = df[['Ur','Ux', 'P']].values #Ux is actually Uz
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle = shuffle)
        X_test = X
        y_test = y
        X_phys = sample_phys_points(X,physics_points_size_ratio)
    return X_phys,X_train, X_test, y_train, y_test

def print_graph(g, indent=''):
    """Prints the computation graph"""
    if g is None:
        return
    print(indent, g)
    for next_g in g.next_functions:
        if next_g[0] is not None:
            print_graph(next_g[0], indent + '  ')

def calc_derivative(func, var):
    return torch.autograd.grad(func, var, grad_outputs=torch.ones_like(func), create_graph=True)[0]

def calc_derivative_finite_diff(u,rz):
        # create grid
        r_unique = torch.unique(rz[:, 0]) # no derivative for torch.unique operation, this leads to failure in backward pass
        z_unique = torch.unique(rz[:, 1])
        u_grid = u.view(len(r_unique), len(z_unique))
        # Compute finite differences
        dudr = (torch.roll(u_grid, shifts=-1, dims=1) - torch.roll(u_grid, shifts=1, dims=1)) / (2 * (r_unique[1] - r_unique[0]))
        dudz = (torch.roll(u_grid, shifts=-1, dims=0) - torch.roll(u_grid, shifts=1, dims=0)) / (2 * (z_unique[1] - z_unique[0]))
        return dudr.view(-1), dudz.view(-1)

def physics_informed_loss(rz, net, constants, Normaliser, finite_difference = False):
    """
    Compute the physics-informed loss for a neural network.

    If there has been physical normalisation, then then the data is non-dimensionalized
    and the loss function is non-dimensionalized. 
    
    If there has been no physical normalisation,
    then the data is still dimensionalized and the loss function is also dimensionalized.
    Args:
        rz (torch.Tensor): Input tensor containing (r, z) coordinates.
        net: The neural network model.
        constants (dict): Dictionary of physical constants.
        Normaliser (list): List of normalizers.
        finite_difference (bool): Whether to use finite difference or automatic differentiation.

    Returns:
        torch.Tensor: The physics-informed loss.
    """
    # unpack constants
    rho = constants["rho"]
    mu = constants["mu"]
    mu_t = constants["mu_t"]    
    U_inf = constants["U_inf"]  # Characteristic velocity
    D = constants["D"]        # Characteristic length
    P = rho * U_inf*U_inf    # Characteristic pressure
    # check if physical normalisation has been applied (i.e. de-dimensionalisation)
    physical_normaliser = any([Normaliser_.physical for Normaliser_ in Normaliser])
    len_normaliser = len(Normaliser)
    if mu_t is None:
        mu_t = rz[:,4]
    if physical_normaliser:
        rz = rz #/ D  # Non-dimensional coordinates
        nu = mu / (rho * U_inf * D)  # Non-dimensional kinematic viscosity
        nu_t = mu_t / (rho * U_inf * D)  # Non-dimensional turbulent kinematic viscosity (eddy viscosity)
    else:
        nu = mu / rho  # kinematic viscosity
        nu_t = mu_t / rho  # turbulent kinematic viscosity (eddy viscosity)

    r = rz[:, 0]
    rz.requires_grad = True
    uvp = net(rz)  # Non-dimensional velocities and pressure

    # unpack predicted values    
    u_r = uvp[:, 0]
    u_z = uvp[:, 1]
    p = uvp[:, 2] # / P

    # Calculate derivatives and second derivatives using functions
    if finite_difference:
        raise NotImplementedError("Finite difference not implemented yet")
        du_r_dr, du_r_dz = calc_derivative_finite_diff(u_r, rz)
        du_z_dr, du_z_dz = calc_derivative_finite_diff(u_z, rz)
        dp_dr, dp_dz = calc_derivative_finite_diff(p, rz)
        # second derivatives
        d2u_r_dr2 = calc_derivative_finite_diff(du_r_dr, rz)[0]
        d2u_r_dz2 = calc_derivative_finite_diff(du_r_dz, rz)[1]
        d2u_z_dr2 = calc_derivative_finite_diff(du_z_dr, rz)[0]
        d2u_z_dz2 = calc_derivative_finite_diff(du_z_dz, rz)[1]
        
    else:
        # Calculate the gradients using automatic differentiation
        du_r = calc_derivative(u_r, rz)
        du_r_dr, du_r_dz = du_r[:, 0], du_r[:, 1]
        du_z = calc_derivative(u_z, rz)
        du_z_dr, du_z_dz = du_z[:, 0], du_z[:, 1]
        dp = calc_derivative(p, rz)
        dp_dr, dp_dz = dp[:, 0], dp[:, 1]
        # second derivatives
        d2u_r_dr2 = calc_derivative(du_r_dr, rz)[:, 0]
        d2u_r_dz2 = calc_derivative(du_r_dz, rz)[:, 1]
        d2u_z_dr2 = calc_derivative(du_z_dr, rz)[:, 0]
        d2u_z_dz2 = calc_derivative(du_z_dz, rz)[:, 1]

    match (physical_normaliser, len_normaliser):
        case (False, 0):
            # Dimensional - No normalisation 
            mass_conservation = du_r_dr + u_r/r + du_z_dz
            # r-momentum
            r_momentum = u_r*du_r_dr + u_z*du_r_dz + 1/rho*dp_dr - (nu + nu_t)*(1/r*du_r_dr + d2u_r_dr2 + d2u_r_dz2 - u_r/r**2)
            # z-momentum
            z_momentum = u_r*du_z_dr + u_z*du_z_dz + 1/rho*dp_dz - (nu + nu_t)*(1/r*du_z_dr + d2u_z_dr2 + d2u_z_dz2)
            # return the loss
            loss = torch.mean(mass_conservation**2 + r_momentum**2 + z_momentum**2)
        case (True, 1):
            # Non-dimensional - no additional normalisation
            mass_conservation = du_r_dr + u_r / r + du_z_dz
            # # r-momentum and z-momentum equations in non-dimensional form
            r_momentum = u_r * du_r_dr + u_z * du_r_dz + dp_dr - \
                         (nu + nu_t) * (1 / r * du_r_dr + d2u_r_dr2 + d2u_r_dz2 - u_r / r**2)
            z_momentum = u_r * du_z_dr + u_z * du_z_dz + dp_dz - \
                         (nu + nu_t) * (1 / r * du_z_dr + d2u_z_dr2 + d2u_z_dz2)
            loss = torch.mean(mass_conservation**2 + r_momentum**2 + z_momentum**2)
        case (True, 2):
            # Non-dimensional - additional normalisation
            n = Normaliser[-1]
            term_1 = n.denorm_u_r(u_r)/n.denorm_r(r) # u_r / r
            term_2 = du_r_dr / (n.dUrtdUr() * n.drdrt()) # du_r_dr
            term_3 = du_z_dz / (n.dUztdUz() * n.dzdzt()) # du_z_dz
            mass_conservation = term_1 + term_2 + term_3
            # second derivatives:
            d2u_r_dz2_ = d2u_r_dz2/ (n.dUrtdUr() * n.dzdzt()**2)
            d2u_r_dr2_ = d2u_r_dr2/ (n.dUrtdUr() * n.drdrt()**2)
            d2u_z_dz2_ = d2u_z_dz2/ (n.dUztdUz() * n.dzdzt()**2)
            d2u_z_dr2_ = d2u_z_dr2/ (n.dUztdUz() * n.drdrt()**2)

            r_momentum = n.denorm_u_r(u_r) * du_r_dr / (n.dUrtdUr() * n.drdrt()) + \
                        n.denorm_u_z(u_z) * du_r_dz / (n.dUrtdUr() * n.dzdzt()) + \
                        dp_dr / (n.dPtdP() * n.drdrt()) - (nu + nu_t) * (1 / n.denorm_r(r) * du_r_dr / (n.dUrtdUr() * n.drdrt()) + d2u_r_dr2_ + d2u_r_dz2_ - n.denorm_u_r(u_r) / n.denorm_r(r)**2)

            z_momentum = n.denorm_u_r(u_r) * du_z_dr / (n.dUztdUz() * n.drdrt()) + \
                        n.denorm_u_z(u_z) * du_z_dz / (n.dUztdUz() * n.dzdzt()) + \
                        dp_dz / (n.dPtdP() * n.dzdzt()) - (nu + nu_t) * (1 / n.denorm_r(r) * du_z_dr / (n.dUztdUz() * n.drdrt()) + d2u_z_dr2_ + d2u_z_dz2_)
            
            loss = torch.mean(mass_conservation**2 + r_momentum**2 + z_momentum**2)
        case (False, 1):
            # Dimensional - additional normalisation
            n = Normaliser[-1]
            term_1 = n.denorm_u_r(u_r)/n.denorm_r(r) # u_r / r
            term_2 = du_r_dr / (n.dUrtdUr() * n.drdrt()) # du_r_dr
            term_3 = du_z_dz / (n.dUztdUz() * n.dzdzt()) # du_z_dz
            mass_conservation = term_1 + term_2 + term_3
            # second derivatives:
            d2u_r_dz2_ = d2u_r_dz2/ (n.dUrtdUr() * n.dzdzt()**2)
            d2u_r_dr2_ = d2u_r_dr2/ (n.dUrtdUr() * n.drdrt()**2)
            d2u_z_dz2_ = d2u_z_dz2/ (n.dUztdUz() * n.dzdzt()**2)
            d2u_z_dr2_ = d2u_z_dr2/ (n.dUztdUz() * n.drdrt()**2)

            r_momentum = n.denorm_u_r(u_r) * du_r_dr / (n.dUrtdUr() * n.drdrt()) + \
                        n.denorm_u_z(u_z) * du_r_dz / (n.dUrtdUr() * n.dzdzt()) + \
                        dp_dr / (n.dPtdP() * n.drdrt())*1/rho - (nu + nu_t) * (1 / n.denorm_r(r) * du_r_dr / (n.dUrtdUr() * n.drdrt()) + d2u_r_dr2_ + d2u_r_dz2_ - n.denorm_u_r(u_r) / n.denorm_r(r)**2)

            z_momentum = n.denorm_u_r(u_r) * du_z_dr / (n.dUztdUz() * n.drdrt()) + \
                        n.denorm_u_z(u_z) * du_z_dz / (n.dUztdUz() * n.dzdzt()) + \
                        dp_dz / (n.dPtdP() * n.dzdzt())*1/rho - (nu + nu_t) * (1 / n.denorm_r(r) * du_z_dr / (n.dUztdUz() * n.drdrt()) + d2u_z_dr2_ + d2u_z_dz2_)
            loss = torch.mean(mass_conservation**2 + r_momentum**2 + z_momentum**2)
    return torch.mean(mass_conservation**2), torch.mean(r_momentum**2), torch.mean(z_momentum**2)

def plot_losses(losses, fig_prefix,output_dir = "Figures"):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(losses["data"])
    ax1.set_title("data loss")
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
    plt.savefig(f"{output_dir}/{fig_prefix}_losses.pdf")
    plt.close()

def plot_heatmaps(X, outputs, y, fig_prefix="",output_dir = "Figures"):

    def plot_single_heatmap(r, z, values, ax, cmap, cbar_label):
        df = pd.DataFrame.from_dict({'r': r, 'z': z, 'values': values})
        pivoted = df.pivot(index="r",columns= "z",values= "values")
        sns.heatmap(pivoted, ax=ax, cmap=cmap, cbar_kws={'label': cbar_label})
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('z')
        ax.set_ylabel('r')

    def plot_all(X, outputs, y, file_suffix):
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # Changed to 3x3 grid
        
        titles = ['Predicted', 'Actual', 'Error']
        data = [outputs, y, np.abs(y - outputs)]
        
        for i, (title, dat) in enumerate(zip(titles, data)):
            # Plot U
            plot_single_heatmap(X[:, 0], X[:, 1], dat[:, 0], axs[i, 0], 'jet', 'U_r values')
            axs[i, 0].set_title(f'{title} U_r')
            
            # Plot V
            plot_single_heatmap(X[:, 0], X[:, 1], dat[:, 1], axs[i, 1], 'jet', 'U_z values')
            axs[i, 1].set_title(f'{title} U_z')
            
            # Plot P
            plot_single_heatmap(X[:, 0], X[:, 1], dat[:, 2], axs[i, 2], 'jet', 'P values')
            axs[i, 2].set_title(f'{title} P')

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{file_suffix}.pdf")  # Changed file name structure for simplification
        plt.close()
    plot_all(X, outputs, y, fig_prefix)

