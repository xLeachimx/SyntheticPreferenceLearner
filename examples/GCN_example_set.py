# File: GCN_example_set.py
# Author: Michael Huelsman
# Created On: 11 Sept 2020
# Purpose:
#   Defines an example set for GCN portfolio learning.

from random import shuffle
# from torch.utils.data import Dataset
from torch_geometric.data import Dataset
from .relation import Relation
import torch

class GCNExampleSet(Dataset):
    # Precond:
    #   labels is the list of possible string labels.
    #
    # Postcond:
    #   Builds a new empty example set for portfolio learning.
    def __init__(self, labels):
        self.examples = []
        self.labels = labels
        self.device = None

    # Precond:
    #   example is a valid Example object.
    #
    # Postcond:
    #   adds an example to the list
    def add_example(self, example):
        self.examples.append(example)

    # Precond:
    #   examples is a valid list of PortfolioExample objects.
    #
    # Postcond:
    #   Adds the examples in the list to the list of examples.
    def add_example_list(self, examples):
        self.examples.extend(examples)

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns a single list with all examples.
    def example_list(self):
        return self.examples

    # Precond:
    #   None.
    #
    # Postcond:
    #   Shuffles the lists, internally, per agent ID
    def shuffle(self):
        shuffle(self.examples)

    # Precond:
    #   example is a valid PortfolioExample object.
    #
    # Postcond:
    #   Turns an example into a structure which can be used for learning a
    #       neural network.
    def prepare_example(self, example):
        inp = example.data
        label = self.labels.index(example.label)
        if self.device is not None:
            inp = inp.to(self.device)
            # label = label.to(self.device)
        return (inp,label)

    def to_dev(self, device):
        self.device = device

    # Precond:
    #   None.
    #
    # Postcond:
    #   Iterates through each example in the set.
    def each(self):
        for ex in self.examples:
            yield ex

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the total number of examples in the set.
    def __len__(self):
        return len(self.examples)

    # Precond:
    #   i is the index of the example ot retrieve.
    #
    # Postcond:
    #   Returns the ith example
    def __getitem__(self,i):
        return self.prepare_example(self.examples[i])

    # Precond:
    #   i is the index of the example ot retrieve.
    #
    # Postcond:
    #   Returns the ith example
    def get(self,i):
        return self.examples[i]

    # Precond:
    #   n is an integer representing the number of folds.
    #
    # Postcond:
    #   Returns an iterator which returns test and validation set.
    def crossvalidation(self,n):
        self.shuffle()
        for i in range(n):
            train = GCNExampleSet(self.labels)
            valid = GCNExampleSet(self.labels)
            count = 0
            # Before the validation slice
            count = int(len(self.examples)/n)
            train.add_example_list(self.examples[0:i*count])
            # The validation slice
            valid.add_example_list(self.examples[i*count+1:(i+1)*count])
            # After the validation slice
            train.add_example_list(self.examples[((i+1)*count)+1:])
            yield (train,valid)
            del train
            del valid
