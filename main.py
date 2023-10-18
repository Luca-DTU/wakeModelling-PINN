import torch
from torch.utils.data import DataLoader
from src import utils, models, normalisers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from omegaconf import OmegaConf
from softadapt import LossWeightedSoftAdapt



def main(csv_path, learning_rate, num_epochs, batch_size, test_size, drop_hub, 
         fig_prefix, network = models.simpleNet, include_physics = False, normaliser = None,shuffle=True,
         constants = {}, remove_dimensionality = False, adaptive_loss_weights = False,
         epochs_to_make_updates = 10, start_adapting_at_epoch = 0):
    
    network = getattr(models, network)
    X_phys,X_train, X_test, y_train, y_test = utils.load_data(csv_path,test_size=test_size, drop_hub=drop_hub, D = constants["D"])
    normaliser = [getattr(normalisers, n) for n in normaliser] # list of classes
    Normaliser = [] # to be list of class instances
    for n in normaliser:
        # init normaliser
        Normaliser_ = n(X_train, y_train, constants)
        # normalise data
        X_train, y_train = Normaliser_.normalise(X_train, y_train)
        X_test, y_test = Normaliser_.normalise(X_test, y_test)
        X_phys, _ = Normaliser_.normalise(X_phys, None)
        Normaliser.append(Normaliser_)
    train_dataset = utils.dataset(X_train, y_train)
    test_dataset = utils.dataset(X_test, y_test)
    X_phys = torch.from_numpy(X_phys).float().to(device)
    if batch_size == -1:
        batch_size = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=len(test_dataset))
    # Define the model
    model = network().to(device)
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # set up losses
    losses = {"data": [], "physics": []}
    adapt_weights = torch.tensor([1,1])
    softadapt_object  = LossWeightedSoftAdapt(beta=0.1, accuracy_order=epochs_to_make_updates)
    # training loop
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) 
            losses["data"].append(loss.item())
            if include_physics:
                if remove_dimensionality:
                    physics_loss = utils.non_dimensionalized_physics_informed_loss(X_phys, model, constants, Normaliser)
                else:
                    physics_loss = utils.physics_informed_loss(X_phys, model, constants)
                losses["physics"].append(physics_loss.item())
                if adaptive_loss_weights:
                    if epoch % epochs_to_make_updates == 0 and epoch >= start_adapting_at_epoch:
                        sample_data,sample_phys = torch.Tensor(losses["data"]), torch.Tensor(losses["physics"])
                        adapt_weights = softadapt_object.get_component_weights(sample_data,sample_phys, verbose = False)
                        print("Adapt weights: ",adapt_weights)
                        print('Epoch: {}, Loss: {:.4f}, Physics loss: {:.8f}'.format(epoch+1, loss.item(), physics_loss.item()))
                else:
                    print('Epoch: {}, Loss: {:.4f}, Physics loss: {:.8f}'.format(epoch+1, loss.item(), physics_loss.item()))
            else:
                physics_loss = 0
                print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
            w_loss = loss/adapt_weights[0] + physics_loss/adapt_weights[1]
            w_loss.backward()
            optimizer.step()
        
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
            for Normaliser_ in Normaliser:
                X, outputs, y = Normaliser_.denormalise(X, outputs, y)
            utils.plot_heatmaps(X, outputs, y, fig_prefix)
            utils.plot_losses(losses, fig_prefix)

if __name__ == '__main__':
    # Load the config file
    config = OmegaConf.to_container(OmegaConf.load("config.yaml"))
    data_config = config["data"]
    training_config = config["training"]
    # Run the main function
    main(**training_config, **data_config)




    
