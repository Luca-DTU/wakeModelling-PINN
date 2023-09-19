import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D scatter plot
# # Open the netCDF file using xarray
# file_path = 'Data\DTU10MW_cart_plus_cyl.nc'  # Replace with your file path
# ds = xr.open_dataset(file_path)
# # Access the dataset
# print(ds)
# # Access variables
# print("Variables:", ds.data_vars.keys())
# # Convert variables to DataFrames
# df_Ux = ds['Ux'].to_dataframe(name='Ux').reset_index()
# df_Ur = ds['Ur'].to_dataframe(name='Ur').reset_index()
# df_Utheta = ds['Utheta'].to_dataframe(name='Utheta').reset_index()

# # Now, each DataFrame contains columns 'x', 'y', 'z', and the corresponding variable data
# print(df_Ux)
# print(df_Ur)
# print(df_Utheta)
df = pd.read_csv('Data/2d_cart.csv')

rho = 1.225 # kg/m^3 (from PyWakeEllipsys docs)
mu = float(1.78406e-05) # kg/m/s (from PyWakeEllipsys docs)
mu_t = 41.60993336515875
k = 2.9400000000001407
Re = (rho*14*178.3)/mu
U_infty = 14
D = 178.3

fig = plt.figure(figsize=(15, 5))
# Plot for U
ax1 = fig.add_subplot(131)
sc1 = ax1.scatter(df['x'], df['y'], c=df['U'], cmap='viridis')
ax1.set_title('U')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
cbar1 = fig.colorbar(sc1, ax=ax1)
cbar1.set_label('U values')

# Plot for V
ax2 = fig.add_subplot(132)
sc2 = ax2.scatter(df['x'], df['y'], c=df['V'], cmap='plasma')
ax2.set_title('V')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
cbar2 = fig.colorbar(sc2, ax=ax2)
cbar2.set_label('V values')

# Plot for P
ax3 = fig.add_subplot(133)
sc3 = ax3.scatter(df['x'], df['y'], c=df['P'], cmap='inferno')
ax3.set_title('P')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
cbar3 = fig.colorbar(sc3, ax=ax3)
cbar3.set_label('P values')
# Adjust layout
plt.tight_layout()
plt.savefig("Figures/2d_cart.pdf")
# Show the plots
plt.show()