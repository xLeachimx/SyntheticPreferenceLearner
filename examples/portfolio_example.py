# File: portfolio_example.py
# Author: Michael Huelsman
# Created On: 11 Sept 2020
# Purpose:
#   Defines an example used for portfolio learning.
# Notes:
#   An example should be immutable for the most part.

class PortfolioExample:
    # Precond:
    #   features is a valid list of floats.
    #   generator is a string indicating the representation used to generate the
    #       example set which generated the feature vector.
    #
    # Postcond:
    #   Builds a new Example object.
    def __init__(self, features, generator):
        self.feature = features
        self.label = generator


    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the feature vector.
    def get_feature(self):
        return self.feature

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the label.
    def get_label(self):
        return self.label

    # Precond:
    #   other is a valid Example object.
    #
    # Postcond:
    #   Returns true if the examples are identical.
    #   ONLY compares alterantives and the relation.
    def __eq__(self,other):
        return self.feature == other.feature and self.label == other.label

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns a string representing the example.
    def to_s(self):
        str_feature = list(map(lambda x: str(x),self.feature))
        str_feature = ';'.join(str_feature)
        return ','.join(str(self.label),str_feature)
