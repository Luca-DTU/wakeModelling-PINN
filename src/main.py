import torch
from torch.utils.data import DataLoader
from src import utils, models 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from omegaconf import OmegaConf



def main(csv_path, learning_rate, num_epochs, batch_size, test_size, drop_hub, 
         fig_prefix, network = models.simpleNet, include_physics = False, normaliser = None,shuffle=True,
         constants = {}, remove_dimensionality = False):
    network = getattr(models, network)
    data, X_train, X_test, y_train, y_test = utils.load_data(csv_path,test_size=test_size, drop_hub=drop_hub, D = constants.D)
    if normaliser is not None:
        normaliser = getattr(utils, normaliser)
        # init normaliser
        Normaliser = normaliser(X_train, y_train, constants)
        # normalise data
        X_train, y_train = Normaliser.normalise(X_train, y_train)
        X_test, y_test = Normaliser.normalise(X_test, y_test)
    train_dataset = utils.dataset(X_train, y_train)
    test_dataset = utils.dataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
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
                if remove_dimensionality:
                    physics_loss = utils.non_dimensionalized_physics_informed_loss(batch_X, model, constants, physical_normaliser=Normaliser.physical)
                else:
                    physics_loss = utils.physics_informed_loss(batch_X, model, constants)
                loss = loss + physics_loss
            loss.backward()
            optimizer.step()
        
        if include_physics: 
            losses["physics"].append(physics_loss.item())
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
            if normaliser is not None:
                X, outputs, y = Normaliser.denormalise(X, outputs, y)
            utils.plot_heatmaps(X, outputs, y, fig_prefix)
            utils.plot_losses(losses, fig_prefix)

if __name__ == '__main__':
    # Load the config file
    config = OmegaConf.load("config.yaml")
    data_config = config.data
    training_config = config.training
    # Run the main function
    main(**training_config, **data_config)




    
