import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader


# Constants.
from models.shallow_relu import ShallowRelu
from utils.plotting import plot_data

device = "cpu"
n = 10

# TODO: start with a "large" learning rate, halve it until loss starts to decrease.
# TODO: implement stopping condition for training: loss(t)- loss(t-1) < 10^{-8}.
# TODO: implement anti-symmetric initialisation.
# TODO: solve variational problem over the domain, calculate infinity norm of difference.


def generate_data():
    # Generate data.
    x = np.concatenate((np.linspace(-2 * np.pi, 0, 10000), np.linspace(np.pi, 3 * np.pi, 10000)))
    x_test = np.linspace(-2 * np.pi, 3 * np.pi, 25_000)
    y = np.sin(x)
    y_test = np.sin(x_test)

    training_data = np.array(list(zip(x, y)))
    test_data = np.array(list(zip(x_test, y_test)))

    # We're doing full-batch gradient descent, so the batch_size = n
    train_dataloader = DataLoader(training_data, batch_size=len(x))
    test_dataloader = DataLoader(test_data, batch_size=len(x_test))
    return train_dataloader, test_dataloader, x_test, y_test


if __name__ == '__main__':
    pl.seed_everything(1337)

    train_dataloader, test_dataloader, x_test, y_test = generate_data()
    model = ShallowRelu(n, 1, 1).to(device).float()

    epochs = 100
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[])
    trainer.fit(model, train_dataloader)

    trainer.test(test_dataloaders=test_dataloader)

    y_pred = model(torch.tensor(x_test).float().unsqueeze(1).to(device))
    plot_data(x_test, y_pred.detach().numpy(), y_test)
