import torch
from torch.utils.data import DataLoader
from src import utils, models 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(csv_path, learning_rate, num_epochs, batch_size, test_size, drop_hub, fig_prefix, network = models.simpleNet, include_physics = False):
    X_train, X_test, y_train, y_test, min_x, max_x, min_y, max_y = utils.load_data(csv_path,test_size=test_size, drop_hub=drop_hub)
    # Create the dataset and dataloader
    train_dataset = utils.two_dim_dataset(X_train, y_train, min_x, max_x, min_y, max_y)
    test_dataset = utils.two_dim_dataset(X_test, y_test, min_x, max_x, min_y, max_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=len(test_dataset))
    # Define the model
    model = network().to(device)
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    losses = {"collocation": [], "physics": []}
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) 
            losses["collocation"].append(loss.item())
            if include_physics:
                physics_loss = utils.physics_informed_loss(batch_X, model)
                losses["physics"].append(physics_loss.item())
                loss = loss + physics_loss
            loss.backward()
            optimizer.step()
        if include_physics:
            print('Epoch: {}, Loss: {:.4f}, Physics loss: {:.4f}'.format(epoch+1, loss.item(), physics_loss.item()))
        else:
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
            utils.plot_heatmaps(X, outputs, y, fig_prefix)

if __name__ == '__main__':
    csv_path = 'Data/2d_cart.csv'
    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 1000
    test_size = 0.99
    drop_hub = True
    fig_prefix = "simplenet"
    main(csv_path, 
        learning_rate,
        num_epochs,
        batch_size, 
        test_size, 
        drop_hub, 
        fig_prefix, 
        include_physics=True)



    
