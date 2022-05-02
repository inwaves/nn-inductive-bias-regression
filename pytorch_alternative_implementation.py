import time

import torch.cuda
import torch.nn as nn
import torch.optim as optim

from models.shallow_relu import PlainTorchAsiShallowRelu
from utils.utils import setup

device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    # Hyperparameters
    num_epochs = 1000
    learning_rate = 1e-3

    train_dataloader, test_dataloader, x_train, y_train, x_test, y_test, args, model = setup()
    model = PlainTorchAsiShallowRelu(args.hidden_units, 1, 1).to(device).float()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    tic = time.time()
    # Train the model
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            inputs, outputs = data[:, 0].float().unsqueeze(1), data[:, 1].float().unsqueeze(1)
            preds = model(inputs)
            loss = criterion(preds, outputs)
            print('Epoch: {} \tLoss: {}'.format(epoch, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    toc = time.time()
    print(f"Training took {toc - tic:.2f} seconds.")


if __name__ == '__main__':
    train()