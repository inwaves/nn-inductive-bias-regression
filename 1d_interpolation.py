import numpy as np
import torch.nn as nn
import torch

from torch.utils.data import DataLoader


# Constants.
from models.shallow_relu import ShallowRelu
from utils.plotting import plot_data

device = "cpu"
n = 10

# TODO: port me to Pytorch Lightning!
# TODO: start with a "large" learning rate, halve it until loss starts to decrease.
# TODO: implement stopping condition for training: loss(t)- loss(t-1) < 10^{-8}.
# TODO: implement anti-symmetric initialisation.
# TODO: solve variational problem over the domain, calculate infinity norm of difference.


def train(dataloader, model, loss_fn, optimiser):
    """Train model for one epoch."""
    size = len(dataloader.dataset)
    model.train()
    for i, batch in enumerate(dataloader):
        x, y = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)

        # Compute prediction error.
        pred = model(x)

        # print(f"pred: {pred}")
        loss = loss_fn(pred, y)

        # Back-propagate error.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        return loss.item()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Sets the model to evaluation mode, no gradients computed.
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
            pred = model(x)
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
    batch_size = len(x)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader, x_test, y_test


if __name__ == '__main__':
    train_dataloader, test_dataloader, x_test, y_test = generate_data()
    model = ShallowRelu(n, 1, 1).to(device).float()
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-5)

    epochs = 500
    for t in range(epochs):
        loss = train(train_dataloader, model, loss_fn, optimiser)
        print(f"Epoch {t + 1:>5d} loss: {loss:>8f}")

    test(test_dataloader, model, loss_fn)

    y_pred = model(torch.tensor(x_test).float().unsqueeze(1).to(device))
    plot_data(x_test, y_pred.detach().numpy(), y_test)
