import os
import pandas as pd
import re
# print all df without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def analyse_multirun(path):
    # Create an empty dataframe to store the test loss values and other parameters
    df = pd.DataFrame(columns=["Test Loss", "Learning Rate", "Batch Size", "Include Physics", "Network", "Physics Points Size Ratio"])

    # Loop through each subfolder in the directory
    for folder in os.listdir(path):
        # Check if the current item is a directory
        if os.path.isdir(os.path.join(path, folder)):
            # Define the path to the main.log file in the current subfolder
            log_path = os.path.join(path, folder, "main.log")
            # Open the log file and read its contents
            with open(log_path, "r") as f:
                log_contents = f.read()
            # Find the line containing the test loss value
            loss_line = [line for line in log_contents.split("\n") if "Test loss:" in line][0]
            # Extract the test loss value from the line
            test_loss = float(loss_line.split(":")[-1])
            
            # Define the path to the overrides.yaml file in the current subfolder
            overrides_path = os.path.join(path, folder, ".hydra", "overrides.yaml")
            # Open the overrides file and read its contents
            with open(overrides_path, "r") as f:
                overrides_contents = f.read()
            # Extract the relevant parameters from the overrides file
            learning_rate = float(re.findall(r"learning_rate=(\d+\.\d+)", overrides_contents)[0])
            batch_size = int(re.findall(r"batch_size=(\d+)", overrides_contents)[0])
            include_physics = bool(re.findall(r"include_physics=(\w+)", overrides_contents)[0] == "True")
            network = re.findall(r"network=(\w+)", overrides_contents)[0]
            physics_points_size_ratio = float(re.findall(r"physics_points_size_ratio=(\d+\.\d+)", overrides_contents)[0])
            
            # Add the test loss value and other parameters to the dataframe
            df.loc[folder] = [test_loss, learning_rate, batch_size, include_physics, network, physics_points_size_ratio]

    # set index as int and sort it
    df.index = df.index.astype(int)
    df = df.sort_index()
    # sort by Test Loss
    df = df.sort_values(by="Test Loss")
    return df
if __name__ == "__main__":
    # Define the path to the directory containing the subfolders
    path = "multirun/2023-11-08/13-38-40"
    df_minmax = analyse_multirun(path)
    print(df_minmax)
    print(df_minmax.iloc[:20].to_latex(index = False,float_format="%.4f"))
    path = "multirun/2023-11-09/11-38-28"
    df_z = analyse_multirun(path)
    print(df_z)
    print(df_z.iloc[:20].to_latex(index = False))
    print("end")