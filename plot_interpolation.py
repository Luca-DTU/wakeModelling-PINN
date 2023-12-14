from src import utils, models, normalisers
import xarray as xr
import pandas as pd
import numpy as np

path = "Data/RANS_1wt_irot_v2.nc"
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
df = df.drop(df[(np.sqrt(df['r']**2 + df['z_cyl']**2) <= 1)].index)
df = df.drop(["U_theta","muT"],axis=1)
df_test = df[df["CT"] == 0.73105] # specific case here, will need to be changed in refactor
df_train = df[df["CT"].isin([0.63388,0.814])]
ti_unique = df_train["TI_amb"].unique()
for ti in ti_unique:
    df_ti = df_train[df_train["TI_amb"] == ti]
    df_ti_low = df_ti[df_ti["CT"] == 0.63388]
    df_ti_high = df_ti[df_ti["CT"] == 0.814]
    ###
    CT_target = 0.73105
    CT_low = df_ti_low['CT'].iloc[0]  # Assuming all CT values are the same in df_ti_low
    CT_high = df_ti_high['CT'].iloc[0]  # Assuming all CT values are the same in df_ti_high

    # Merge the two dataframes on 'z_cyl' and 'r'
    df_merged = pd.merge(df_ti_low, df_ti_high, on=['z_cyl', 'r'], suffixes=('_low', '_high'))

    # Calculating the interpolation factor
    factor = (CT_target - CT_low) / (CT_high - CT_low)

    # Columns to interpolate
    columns_to_interpolate = ["U_r", "U_z", "P"]

    # Creating an empty dataframe for interpolated values
    df_interpolated = pd.DataFrame()

    # Copying 'z_cyl' and 'r'
    df_interpolated[['r','z_cyl']] = df_merged[['r','z_cyl']]

    # Interpolating other columns
    for col in columns_to_interpolate:
        df_interpolated[col] = df_merged[col + '_low'] + factor * (df_merged[col + '_high'] - df_merged[col + '_low'])

    df_interpolated["TI_amb"] = ti

    X = df_interpolated[['r','z_cyl']].to_numpy()
    outputs = df_interpolated[['U_r','U_z','P']].to_numpy()
    y = df_interpolated.merge(df_test, on=['z_cyl', 'r', 'TI_amb'], suffixes=('', '_actual'))
    y = y[['U_r_actual','U_z_actual','P_actual']].to_numpy()
    utils.plot_heatmaps(X,outputs,y,fig_prefix=f"interpolation_{ti}",output_dir="Figures/interpolation")

