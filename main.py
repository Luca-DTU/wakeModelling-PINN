import torch
from torch.utils.data import DataLoader
from src import utils, models, normalisers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import hydra
from omegaconf import OmegaConf
from softadapt import LossWeightedSoftAdapt



def main(csv_path, learning_rate, num_epochs, batch_size, test_size, drop_hub, 
         fig_prefix, network = models.simpleNet, include_physics = False, normaliser = None,shuffle=True,
         constants = {}, adaptive_loss_weights = False,
         epochs_to_make_updates = 10, start_adapting_at_epoch = 0, finite_difference = False,
         seed = 42, physics_points_size_ratio = 1.0):
    torch.manual_seed(seed)
    network = getattr(models, network)
    X_phys,X_train, X_test, y_train, y_test = utils.load_data(csv_path,
                                                            test_size=test_size,
                                                            drop_hub=drop_hub,
                                                            D = constants["D"],
                                                            shuffle=shuffle,
                                                            random_state=seed,
                                                            physics_points_size_ratio = physics_points_size_ratio)
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
    phys_batch_size = X_phys.size()[0]//len(train_loader)
    for epoch in range(num_epochs):
        epoch_losses = {"data": [], "physics": []}
        for n_batch, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_phys = X_phys[n_batch*phys_batch_size:(n_batch+1)*phys_batch_size,:].to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) 
            epoch_losses["data"].append(loss.item())
            if include_physics:
                mass_conservation, r_momentum, z_momentum = utils.physics_informed_loss(batch_phys, model, constants, Normaliser, finite_difference)
                physics_loss = mass_conservation + r_momentum + z_momentum
                epoch_losses["physics"].append(physics_loss.item())
            else:
                epoch_losses["physics"].append(0)
            weighted_loss = loss/adapt_weights[0] + physics_loss/adapt_weights[1]
            weighted_loss.backward()
            optimizer.step()
        losses["data"].append(sum(epoch_losses["data"])/len(epoch_losses["data"]))
        losses["physics"].append(sum(epoch_losses["physics"])/len(epoch_losses["physics"]))
        if epoch % 50 == 0:
            print('Epoch: {}, Loss: {:.4f}, Physics loss: {:.8f}'.format(epoch+1, losses["data"][-1], losses["physics"][-1]))
        if adaptive_loss_weights:
            if epoch % epochs_to_make_updates == 0 and epoch >= start_adapting_at_epoch:
                sample_data,sample_phys = torch.Tensor(losses["data"]), torch.Tensor(losses["physics"])
                adapt_weights = softadapt_object.get_component_weights(sample_data,sample_phys, verbose = False)
                print("Adapt weights: ",adapt_weights)

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
            utils.plot_heatmaps(X, outputs, y, fig_prefix) # this is too slow
            utils.plot_losses(losses, fig_prefix)

@hydra.main(config_path="conf", config_name="config",version_base=None)
def my_app(config):
    data_config = config["data"]
    training_config = config["training"]
    # Run the main function
    main(**training_config, **data_config)

if __name__ == '__main__':
    my_app()





    
