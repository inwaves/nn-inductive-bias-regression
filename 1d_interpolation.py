import numpy as np
import torch.nn as nn
import torch

from torch.utils.data import DataLoader


# Constants.
from models.shallow_relu import ShallowRelu

device = "cpu"
n = 512


def train(dataloader, model, loss_fn, optimiser):
    """Train model for one epoch."""
    size = len(dataloader.dataset)
    model.train()
    for i, batch in enumerate(dataloader):
        X, y = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)

        # Compute prediction error.
        pred = model(X)

        # print(f"pred: {pred}")
        loss = loss_fn(pred, y)

        # Backpropagate error.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if i % 100 == 0:
            loss, current = loss.item(), i * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Sets the model in evaluation mode, no gradients computed.
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            X, y = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100 * correct):>0.1f}%, avg. loss: {test_loss:>8f}\n")


def generate_data():
    # Generate data.
    x = np.concatenate((np.linspace(-2 * np.pi, 0, 10000), np.linspace(np.pi, 3 * np.pi, 10000)))
    x_test = np.linspace(-2 * np.pi, 3 * np.pi, 25_000)
    y = np.sin(x)
    y_test = np.sin(x_test)

    training_data = np.array(list(zip(x, y)))
    test_data = np.array(list(zip(x_test, y_test)))

    # Split data into batches for training.
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader = generate_data()
    model = ShallowRelu(n).to(device).float()
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n---------------------------------")
        train(train_dataloader, model, loss_fn, optimiser)
        test(test_dataloader, model, loss_fn)
    print("Done!")
