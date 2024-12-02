# Import libraries 
import torch
from torch import nn
import torch.optim as optim


# Define training tools
loss_functions = {
    "binary_cross_entropy": nn.BCELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "mse_loss": nn.MSELoss,
}
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}
# Search for a device
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# Define parameters configuration
params = {
    "data_splitting" : {
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
    },
    
    "dataloader": {
        "batch_size": 16,
        "shuffle": True,
        "num_workers" : 0
    },
    
    "loss": {
        "type": "mse_loss",
        "params" : {
            "reduction": "mean"
        }
    },

    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": (0.9, 0.999), 
            "eps": 1e-08,
            "weight_decay": 0.001,
            "momentum": 0,
            "weight_decay": 0.001,
        }
    },

    "training": {
        "epochs" : 100
    },
}  