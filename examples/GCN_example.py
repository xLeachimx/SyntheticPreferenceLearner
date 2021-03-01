# File: GCN_example.py
# Author: Michael Huelsman
# Created On: 11 Sept 2020
# Purpose:
#   A class for using a preference graph with a GCN.

from .relation import Relation
from torch_geometric.data import Data
import torch

class GCNExample:
    # Precond:
    #   ex_set is a valid instance of the ExampleSet class.
    #   label is a string representing the type of language used to generate
    #       the example set.
    #
    # Postcond:
    #   Builds a new example from the given example set and label
    def __init__(self, ex_set, label):
        self.data = None
        self.label = None
        self.load_example_set(ex_set, label)

    # Precond:
    #   ex_set is a valid instance of the ExampleSet class.
    #   label is a string representing the type of language used to generate
    #       the example set.
    #
    # Postcond:
    #   Replaces the example graph with one generated base on the example set.
    def load_example_set(self, ex_set, label):
        nodes = []
        edges = []
        for ex in ex_set.each():
            alts = ex.get_alts()
            alts = list(map(lambda x: x.to_list(),alts))
            # Generate set of nodes.
            if not (alts[0] in nodes):
                nodes.append(alts[0])
            if not (alts[1] in nodes):
                nodes.append(alts[1])
            alt_indices = list(map(lambda x: nodes.index(x),alts))
            # Generate set of edges.
            if ex.get_relation() == Relation.strict_dispreference():
                edges.append(alt_indices[::-1])
            elif ex.get_relation() == Relation.dispreference():
                edges.append(alt_indices[::-1])
            elif ex.get_relation() == Relation.equal():
                edges.append(alt_indices[::-1])
                edges.append(alt_indices)
            elif ex.get_relation() == Relation.preference():
                edges.append(alt_indices)
            elif ex.get_relation() == Relation.strict_preference():
                edges.append(alt_indices)
            # Convert to tensors
            node_tensor = torch.tensor(nodes, dtype=torch.float)
            # Solve problem with empty edge list
            if len(edges) == 0:
                edges.append([0,0])
            edge_tensor = torch.tensor(edges, dtype=torch.long)
            self.data = Data(x=node_tensor, edge_index=edge_tensor.t().contiguous())
            self.label = label

    def to(self, device):
        self.data = self.data.to(device)
