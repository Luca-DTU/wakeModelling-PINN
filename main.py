import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
# Define the model
class simpleNet(torch.nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.layer1 = torch.nn.Linear(2, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
    
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

def main(csv_path, learning_rate, num_epochs, batch_size, test_size, drop_hub, fig_prefix, network = simpleNet):
    X_train, X_test, y_train, y_test, min_x, max_x, min_y, max_y = load_data(csv_path,test_size=test_size, drop_hub=drop_hub)
    # Create the dataset and dataloader
    train_dataset = two_dim_dataset(X_train, y_train, min_x, max_x, min_y, max_y)
    test_dataset = two_dim_dataset(X_test, y_test, min_x, max_x, min_y, max_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=len(test_dataset))
    # Define the model
    model = network().to(device)
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
    # Test the model
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            print('Test loss: {:.4f}'.format(loss.item()))
            # plot the results
            X = X.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            # Denormalise
            X = X*(max_x - min_x) + min_x
            outputs = outputs*(max_y - min_y) + min_y
            y = y*(max_y - min_y) + min_y
            plot(X, outputs, y, fig_prefix)

if __name__ == '__main__':
    csv_path = 'Data/2d_cart.csv'
    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 64 
    test_size = 0.99
    drop_hub = True
    fig_prefix = "simplenet"
    main(csv_path, learning_rate, num_epochs, batch_size, test_size, drop_hub, fig_prefix)



    
