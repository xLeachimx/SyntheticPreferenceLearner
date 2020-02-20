# File: lp_tree.py
# Author: Michael Huelsman
# Created On: 29 Jan 2019
# Purpose:
#   Specifies a class for handling LP-tree Models.

# Class for handling condition checking.
class LPTree_Condition:
    # Precond:
    #   alt is a valid Alternative object.
    #   conds is a list of attributes.
    #
    # Postcond:
    #   Build a new LPTree_Condition object.
    def __init__(self, alt, conds):
        self.condition = {}
        for attr in conds:
            self.condition[attr] = alt.value(attr)

    # Precond:
    #   alt is a valid Alternative object.
    #
    # Postcond:
    #   Returns true if the condition and alternative match.
    def match(self, alt):
        for attr in self.condition:
            if alt[attr] != self.condition[attr]:
                return False
        retrun True

# Class for handling the order of an attribute domain in an LPTree.
class LPTree_Order:
    # Precond:
    #   attr is the attrbute that the order is associated with.
    #
    # Postcond:
    #   Builds an empty LPTree order
    def __init__(self, attr):
        self.attr = attr
        self.pairs = []

    # Precond:
    #   alt is a valid Alternative object.
    #
    # Postcond:
    #   Returns the order for the attribute given the alternative.
    def get(self,alt):
        for pair in pairs:
            if pair[0].match(alt):
                return pair[1]

    # Precond:
    #   condition is a valid LPTree_Condition object.
    #   order is an ordered list of integer values.
    #
    # Postcond:
    #   Adds the condition order pair.
    def add_pair(condition, order):
        self.pairs.append((condition,order))

# Class for handling a tree node.
class LPTree_Node:
    # Precond:
    #   attr is the attribute associated with the node.
    #   order is a valid LPTree_Order object.
    #   branch is true if the Node branches.
    #   domain is a valid Domain object.
    #
    # Postcond:
    #   Builds an empty LPTree_Node for the given attribute.
    def __init__(self, attr, order, branch, domain):
        self.attr = attr
        self.order = order
        self.branch = branch
        self.children = []
        self.domain = domain

    # Precond:
    #   alt1 is a valid Alternative object.
    #   alt2 is a valid Alternative object.
    def compare(self,alt1,alt2):
        # retrieve the
        val_order = self.order.get(alt1)


class LPTree:
    # Precond:
    #   domain is a valid Domain object.
    #   c_pref is a boolean indicating if conditional preference is possible.
    #   c_import is a boolean indicating if conditional importance is possible.
    #
    # Postcond:
    #   Produces an new empty LPTree for the given domain.
    def __init__(self,domain):
