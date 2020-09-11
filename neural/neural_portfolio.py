# File: neural_portfolio.py
# Author: Michael Huelsman
# Created On: 11 Sept 2020
# Purpose:
#   A neural net for learning which features best align with which preference
#   representations.

import torch
import torch.nn as nn
# import torch.nn.function as F
import torch.optim as optim
from torch.utils.data import DataLoader

class NeuralPortfolio(nn.Module):
    def __init__(self, size):
        super(NeuralPortfolio, self).__init__()
        self.layers = [nn.Linear(size[i],size[i+1]) for i in range(len(size)-1)]
        self.layers = nn.ModuleList(self.layers)
        self.internal = nn.ReLU()
        self.squash = nn.LogSoftmax(dim=0)

    def forward(self, x):
        for i in range(len(self.layers)):
            if i != 0:
                x = self.internal(x)
            x = self.layers[i](x)
        return x

    def forward_squash(self, x):
        for i in range(len(self.layers)):
            if i != 0:
                x = self.internal(x)
            x = self.layers[i](x)
        x = self.squash(x)
        return x

# Precond:
#   ex_set is a valid ExampleSet object.
#   layers is a list of integers indicating the layer sizes.
#   epochs is an integer indicating the number of epochs to train for.
#   device is the device where the learner should reside.
#
# Postcond:
#   Trains and returns a neural net which has learned off the given example set.
def train_neural_portfolio(ex_set, layers, epochs, device=None):
    result = NeuralPortfolio(layers)
    if device is not None:
        result = result.to(device)
    # criterion = nn.L1Loss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(result.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=100)
    for epoch in range(epochs):
        globalLoss = 0.0
        for examples in DataLoader(dataset=ex_set, batch_size=10, pin_memory=True):
            inps = labels = None
            if device is not None:
                inps, labels = examples[0].to(device), examples[1].to(device)
            else:
                inps, labels = examples
            optimizer.zero_grad()
            outputs = result(inps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            globalLoss += loss.item()
            del inps
            del loss
            del labels
            del outputs
            del examples
        scheduler.step(globalLoss)
        # Debug line
        print(epoch,"->",globalLoss)
    del ex_set
    del optimizer
    del scheduler
    del criterion
    return result
