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
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

class NeuralPortfolioGCN(nn.Module):
    def __init__(self, conv_size, lin_size):
        super(NeuralPortfolioGCN, self).__init__()
        self.convs = [GCNConv(conv_size[i],conv_size[i+1]) for i in range(len(conv_size)-1)]
        self.layers = [nn.Linear(lin_size[i],lin_size[i+1]) for i in range(len(lin_size)-1)]
        self.conv_length = len(self.convs)
        self.lin_length = len(self.layers)
        self.layers = nn.ModuleList(self.convs+self.layers)
        self.internal = nn.ReLU()
        self.squash = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(len(self.layers)):
            if i != 0:
                x = self.internal(x)
            if i < self.conv_length:
                x = self.layers[i](x, edge_index)
            else:
                x = self.layers[i](x)
        x = global_mean_pool(x, batch)
        return x

    def forward_squash(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.layers)):
            if i != 0:
                x = self.internal(x)
            if i < self.conv_length:
                x = self.layers[i](x, edge_index)
            else:
                x = self.layers[i](x)
        batch = torch.zeros((x.size()[0]),dtype=torch.long).to(x.device)
        x = global_mean_pool(x, batch)
        return self.squash(x)

# Precond:
#   ex_set is a valid ExampleSet object.
#   layers is a list of integers indicating the layer sizes.
#   epochs is an integer indicating the number of epochs to train for.
#   device is the device where the learner should reside.
#
# Postcond:
#   Trains and returns a neural net which has learned off the given example set.
def train_neural_portfolio_gcn(ex_set, conv_layers, lin_layers, epochs, device=None):
    result = NeuralPortfolioGCN(conv_layers, lin_layers)
    if device is not None:
        result = result.to(device)
        ex_set.to_dev(device)
    # criterion = nn.L1Loss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.ASGD(result.parameters(), lr=0.01)
    # optimizer = optim.Adam(result.parameters(), lr=0.00005)
    # optimizer = optim.Adagrad(result.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1000)
    for epoch in range(epochs):
        globalLoss = 0.0
        for examples in DataLoader(dataset=ex_set, batch_size=5, shuffle=True):
            inps = labels = None
            inps, labels = examples
            if device is not None:
                labels = labels.to(device)
            # if device is not None:
            #     inps, labels = examples[0].to(device), examples[1].to(device)
            # else:
            #     inps, labels = examples
            optimizer.zero_grad()
            outputs = result(inps)
            # print(outputs)
            # print(labels)
            # print(outputs, labels)
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
