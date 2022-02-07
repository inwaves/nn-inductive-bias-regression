import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


# Generate data.
x = np.concatenate((np.linspace(-2*np.pi, 0, 100), np.linspace(np.pi, 3*np.pi, 100)))
y = np.sin(x)


# Plot data points.
fig, ax = plt.subplots()
ax.plot(x, y)


# Training on CPU.
device = "cpu"

# Specify a model.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.linear_relu_stack(x)
    
model = NeuralNetwork().to(device)
print(model)


def train(dataloader, model, loss_fn, optimiser):
    """Train model for one epoch."""
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # What is this boilerplate?
        
        # Compute prediction error.
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagate error.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Sets the model in evaluation mode, no gradients computed.
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}get_ipython().run_line_magic(",", " avg. loss: {test_loss:>8f}\n\")")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------------")
    train(train_dataloader, model, loss_fn, optimiser)
    test(test_dataloader, model, loss_fn)
print("Doneget_ipython().getoutput("")")


# Plot the fitted curve.


# Vary hyperparameters.



