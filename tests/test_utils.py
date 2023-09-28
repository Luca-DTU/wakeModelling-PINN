import torch
from src.utils import normalised_dataset, dataset, load_data, plot_losses, plot_heatmaps, physics_informed_loss
import pytest

def test_normalised_dataset():
    X = torch.randn(100, 2)
    y = torch.randn(100, 1)
    min_x = torch.tensor([-1.0, -1.0])
    max_x = torch.tensor([1.0, 1.0])
    min_y = torch.tensor([-1.0])
    max_y = torch.tensor([1.0])
    dataset = normalised_dataset(X, y, min_x, max_x, min_y, max_y)
    assert len(dataset) == 100
    assert dataset[0][0].shape == torch.Size([2])
    assert dataset[0][1].shape == torch.Size([1])

def test_dataset():
    X = torch.randn(100, 2)
    y = torch.randn(100, 1)
    dataset = dataset(X, y)
    assert len(dataset) == 100
    assert dataset[0][0].shape == torch.Size([2])
    assert dataset[0][1].shape == torch.Size([1])

def test_load_data():
    csv_path = "data.csv"
    X_train, X_test, y_train, y_test = load_data(csv_path, test_size=0.2, random_state=42, drop_hub=True)
    assert X_train.shape == torch.Size([80, 2])
    assert X_test.shape == torch.Size([20, 2])
    assert y_train.shape == torch.Size([80, 1])
    assert y_test.shape == torch.Size([20, 1])

def test_physics_informed_loss():
    rz = torch.randn(100, 2)
    net = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 3))
    loss = physics_informed_loss(rz, net)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0.0

def test_plot_losses():
    losses = {"collocation": [0.1, 0.05, 0.01], "physics": [0.2, 0.1, 0.05]}
    fig_prefix = "test"
    plot_losses(losses, fig_prefix)

def test_plot_heatmaps():
    X = torch.randn(100, 2)
    outputs = torch.randn(100, 3)
    y = torch.randn(100, 3)
    fig_prefix = "test"
    plot_heatmaps(X, outputs, y, fig_prefix)