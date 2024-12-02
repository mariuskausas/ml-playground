# Import libraries
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor

# Load the training data
mnist_train = datasets.MNIST(
    root="datasets",
    train=True,
    download=True,
    transform=Compose([
        ToTensor(),
        torch.flatten,
    ]))

# Load the test data
mnist_test = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=Compose([
        ToTensor(),
        torch.flatten,
    ]))

# Generate indices: instead of the actual data we pass in integers instead
train_indices, val_indices, _, _ = train_test_split(
    range(len(mnist_train)),
    mnist_train.targets,
    stratify=mnist_train.targets,
    test_size=0.1,
)

# Generate a subset based on indices
training_dataset = Subset(mnist_train, train_indices)
validation_dataset = Subset(mnist_train, val_indices)

# Create generators
training_generator = DataLoader(training_dataset)
validation_generator = DataLoader(validation_dataset)
test_generator = DataLoader(mnist_test)
