# File: GCN_example_full.py
# Author: Michael Huelsman
# Created On: 15 Jan 2021
# Purpose:
#   A class for using a preference graph with a GCN. Designed to speed up
#   generating full examples.

from .relation import Relation
from torch_geometric.data import Data
import torch

class GCNExampleFull:
    # Precond:
    #   domain is a valid Domain object.
    #   agent is a valid Agent object.
    #   label is a string representing the type of language used to generate
    #       the example set.
    #
    # Postcond:
    #   Builds a new example from the given example set and label
    def __init__(self, domain, agent, label):
        self.data = None
        self.label = None
        self.build_graph(domain, agent, self.label)

    # Precond:
    #   domain is a valid Domain object.
    #   agent is a valid Agent object.
    #   label is a string representing the type of language used to generate
    #       the example set.
    #
    # Postcond:
    #   Builds the example graph from the given agent over the given domain.
    def build_graph(self, domain, agent, label):
        nodes = [item for item in domain.each()]
        edges = []
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                relation = agent.relation(nodes[i],nodes[j])
                if relation == Relation.strict_dispreference():
                    edges.append((j,i))
                elif relation == Relation.dispreference():
                    edges.append((j,i))
                elif relation == Relation.equal():
                    edges.append((j,i))
                    edges.append((i,j))
                elif relation == Relation.preference():
                    edges.append((i,j))
                elif relation == Relation.strict_preference():
                    edges.append((i,j))
        nodes = list(map(lambda x: x.to_list(),nodes))
        node_tensor = torch.tensor(nodes, dtype=torch.float)
        if len(edges) == 0:
            edges.append([0,0])
        edge_tensor = torch.tensor(edges, dtype=torch.long)
        self.data = Data(x=node_tensor, edge_index=edge_tensor.t().contiguous())
        self.label = label

    def to(self, device):
        self.data = self.data.to(device)
