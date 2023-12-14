import torch
from torch.utils.data import DataLoader
from src import utils, models, normalisers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import hydra
from omegaconf import OmegaConf
from softadapt import LossWeightedSoftAdapt
import os
import shutil
import logging
import copy
log = logging.getLogger(__name__)

def main(path, learning_rate, num_epochs, batch_size, test_size, drop_hub, 
         fig_prefix, network = models.simpleNet, include_physics = False, normaliser = None,shuffle=True,
         constants = {}, adaptive_loss_weights = False,
         epochs_to_make_updates = 10, start_adapting_at_epoch = 0, 
         seed = 42, physics_points_size_ratio = 1.0, dynamic_collocation = False):
    stored_nn = None
    torch.manual_seed(seed)
    network = getattr(models, network)
    X_phys,X_train, X_test, y_train, y_test = utils.load_data(path,
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
        Normaliser_ = n(X_train[:,:2], y_train, constants)
        # normalise data
        X_train[:,:2], y_train = Normaliser_.normalise(X_train[:,:2], y_train)
        X_test[:,:2], y_test = Normaliser_.normalise(X_test[:,:2], y_test)
        X_phys[:,:2], _ = Normaliser_.normalise(X_phys[:,:2], None)
        Normaliser.append(Normaliser_)
    train_dataset = utils.dataset(X_train, y_train,batch_size=batch_size)
    test_dataset = utils.test_dataset(X_test, y_test)
    X_phys = torch.from_numpy(X_phys).float().to(device)
    if batch_size == -1:
        batch_size = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0) 
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1) # 1 flow case at a time
    # Define the model
    model = network().to(device)
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # set up losses
    losses = {"data": [], "physics": []}
    adapt_weights = torch.tensor([1,1])
    softadapt_object  = LossWeightedSoftAdapt(beta=0.1, accuracy_order=10)
    # training loop
    phys_batch_size = X_phys.size()[0]//len(train_loader)
    log.info(f"number of batches: {len(train_loader)}")
    log.info(f"Physics batch size: {phys_batch_size}")
    for epoch in range(num_epochs):
        epoch_losses = {"data": [], "physics": []}
        for n_batch, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.squeeze(), batch_y.squeeze()
            constants["mu_t"] = torch.unique(batch_X[:,4]).item()
            print(constants["mu_t"])
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_phys = X_phys[n_batch*phys_batch_size:(n_batch+1)*phys_batch_size,:].to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) 
            epoch_losses["data"].append(loss.item())
            if include_physics and epoch >= start_adapting_at_epoch: # despite the names, these quantitites are the sum of squared residuals of the equations
                mass_conservation, r_momentum, z_momentum = utils.physics_informed_loss(batch_phys, model, constants, Normaliser)
                log.debug(f"mass_conservation: {mass_conservation.item()}, r_momentum: {r_momentum.item()}, z_momentum: {z_momentum.item()}")
                physics_loss = mass_conservation + r_momentum + z_momentum
                epoch_losses["physics"].append(physics_loss.item())
            else:
                physics_loss = torch.tensor(0)
                epoch_losses["physics"].append(0)
            weighted_loss = loss/adapt_weights[0] + physics_loss/adapt_weights[1]
            weighted_loss.backward()
            optimizer.step()
        losses["data"].append(sum(epoch_losses["data"]))
        losses["physics"].append(sum(epoch_losses["physics"]))
        if epoch % epochs_to_make_updates == 0:
            log.info('Epoch: {}, Loss: {:.4f}, Physics loss: {:.8f}'.format(epoch+1, losses["data"][-1], losses["physics"][-1]))
            if adaptive_loss_weights and epoch >= (start_adapting_at_epoch+epochs_to_make_updates) and include_physics:
                sample_data,sample_phys = torch.Tensor(losses["data"]), torch.Tensor(losses["physics"])
                relevant_epochs = epoch-start_adapting_at_epoch # epochs with both physics and data loss
                adapt_weights = softadapt_object.get_component_weights(sample_data[-relevant_epochs:],sample_phys[-relevant_epochs:], verbose = False)
                log.info(f"Adapt weights: {adapt_weights}")
            if dynamic_collocation and epoch >= (start_adapting_at_epoch+epochs_to_make_updates) and include_physics:
                if stored_nn is not None:
                    old_outputs = stored_nn(X_phys)
                    new_outputs = model(X_phys)
                    diff = torch.abs(torch.sum(old_outputs-new_outputs,axis=1))
                    # concatenate X_phys and diff
                    X_mat = torch.cat((X_phys,diff.unsqueeze(1)),axis=1).detach().numpy()
                    n_samples = len(X_phys)
                    D = constants["D"] if drop_hub else 0
                    X_phys = utils.sample_points(X_mat, n_samples,Normaliser,D=D,grid_size=100)
                stored_nn = copy.deepcopy(model)


    # Test the model
    model.eval()
    with torch.no_grad():
        for n, (X, y) in enumerate(test_loader):
            X = X.squeeze().to(device)
            y = y.squeeze().to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            log.info('Test loss: {:.4f}'.format(loss.item()))
            
            # plot the results
            X = X.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            for Normaliser_ in Normaliser[::-1]:
                X[:,:2], outputs, y = Normaliser_.denormalise(X[:,:2], outputs, y)
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            prefix = f"{fig_prefix}_{n}"
            utils.plot_heatmaps(X, outputs, y,prefix,output_dir) # this is too slow
        utils.plot_losses(losses, fig_prefix,output_dir)
    # Store model
    torch.save(model.state_dict(), os.path.join(output_dir, f"{fig_prefix}_model.pth"))
    log.info(f"Model saved to {os.path.join(output_dir, f'{fig_prefix}_model.pth')}")
    return loss.item()

@hydra.main(config_path="conf", config_name="big_data",version_base=None)
def my_app(config):
    data_config = config["data"]
    training_config = config["training"]
    # Run the main function
    log.info(f"Running with config: {OmegaConf.to_yaml(config['training'])}")
    # try:
    test_loss = main(**training_config, **data_config)
    # except Exception as e:
    #     print("-----------------------------------")
    #     print("JOB FAILED --- EXCEPTION")
    #     log.error(f"Exception: {e}")
    #     print("CONFIGURATION")
    #     print(f"Running with config: {OmegaConf.to_yaml(config['training'])}")
    #     print("-----------------------------------")
    #     test_loss = 1e10
    return test_loss

def clean_up_empty_files():
    outputs_folder = "outputs"
    for root, dirs, files in os.walk(outputs_folder):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if os.path.isdir(dir_path):
                subdirs = os.listdir(dir_path)
                if len(subdirs) == 2 and "main.log" in subdirs and ".hydra" in subdirs:
                    shutil.rmtree(dir_path)

if __name__ == '__main__':
    clean_up_empty_files()
    my_app()
    





    
