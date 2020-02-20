# File: lp_tree.py
# Author: Michael Huelsman
# Created On: 29 Jan 2019
# Purpose:
#   Specifies a class for handling LP-tree Models.

from ..utility.conditional_preferences import CPT

# Class for handling an LPTNode.
class LPTNode:
    # Precond:
    #   leaf is a boolean that indicates if this node is a leaf.
    #
    # Postcond:
    #   Builds an empty LPTNode.
    def __init__(leaf = False):
        self.leaf = False
        self.attr = None
        self.cpt = None
        self.parents = []

# Class for handling an entire LP-tree.
class LPTree:
    # Precond:
    #   domain is a valid domain object.
    #
    # Postcond:
    #   Builds an empty LP-tree.
    def __init__(self, domain):
        self.domain = domain
        self.root = None
