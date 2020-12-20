# File: neural_portfolio_gcn.py
# Author: Michael Huelsman
# Created On: 11 Sept 2020
# Purpose:
#   A GCN neural net for learning which features best align with which preference
#   representations.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv

class NeuralPortfolioGCN(nn.Module):
    def __init__(self, size):
        super(NeuralPortfolioGCN, self).__init__()
        self.convs = [GCNConv(size[i],size[i+1]) for i in range(len(size)-1)]
        self.convs = nn.ModuleList(self.convs)
        self.internal = nn.ReLU()
        self.squash = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.convs)):
            if i != 0:
                x = self.internal(x)
            x = self.convs[i](x,edge_index)
        return self.squash(x)

    def forward_squash(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.convs)):
            if i != 0:
                x = self.internal(x)
            x = self.convs[i](x,edge_index)
        return self.squash(x)

# Precond:
#   ex_set is a valid ExampleSet object.
#   layers is a list of integers indicating the layer sizes.
#   epochs is an integer indicating the number of epochs to train for.
#   device is the device where the learner should reside.
#
# Postcond:
#   Trains and returns a neural net which has learned off the given example set.
def train_neural_portfolio_gcn(ex_set, layers, epochs, device=None):
    result = NeuralPortfolioGCN(layers)
    if device is not None:
        result = result.to(device)
    # criterion = nn.L1Loss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.ASGD(result.parameters(), lr=0.001)
    optimizer = optim.Adam(result.parameters(), lr=0.001)
    # optimizer = optim.Adagrad(result.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1000)
    for epoch in range(epochs):
        globalLoss = 0.0
        for examples in DataLoader(dataset=ex_set, batch_size=5, pin_memory=True):
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
        # print(epoch,"->",globalLoss)
    del ex_set
    del optimizer
    del scheduler
    del criterion
    return result
