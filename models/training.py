# Import libraries 
from tqdm import tqdm
import torch 
import torch.nn.functional as F

import matplotlib.pyplot as plt


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """ Perform a training loop."""
    running_loss = 0
    
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        
        # Send the batch samples to a device
        X, y = X.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        running_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        # Update model weights
        optimizer.step()
        # Re-set gradients to zero for next batch
        optimizer.zero_grad()

    avg_training_loss = running_loss / len(dataloader)
    
    return avg_training_loss


def validation_loop(dataloader, model, loss_fn, device):
    """ Perform a validation loop."""
    running_loss = 0
    
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:

            # Send the batch samples to a device
            X, y = X.to(device), y.to(device)
            
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            running_loss += loss.item()

    avg_validation_loss = running_loss / len(dataloader)

    return avg_validation_loss


def train_model(training_generator, validation_generator, model, loss_fn, optimizer, epochs):
    
    # Collect results
    training_loss = []
    validation_loss = []
    
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        # Perform a training step
        running_training_loss = train_loop(
            dataloader=training_generator, 
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
        training_loss.append(running_training_loss)

        # Perform a validation step
        running_validation_loss = validation_loop(
            dataloader=validation_generator,
            model=model,
            loss_fn=loss_fn
        )
        validation_loss.append(running_validation_loss)

    return training_loss, validation_loss


def plot_loss(training_loss, validation_loss, model_id):
    """ Plot training and validation losses."""
    fig = plt.figure(figsize=[5, 3])
    ax = fig.add_subplot(111)
    ax.plot(training_loss, label="Training loss")
    ax.plot(validation_loss, label="Validation loss")
    ax.set_title("Model: {}".format(model_id))
    ax.legend()


def test_model_classification(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_model_regression(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output)
            test_loss += F.mse_loss(output, target)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss / (i + 1)))
    