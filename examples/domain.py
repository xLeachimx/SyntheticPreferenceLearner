# File: domain.py
# Author: Michael Huelsman
# Created On: 23 Oct 2019
# Purpose:
#   Defines a combinatorial domain.
# Notes:
#

from alternative import Alternative

class Domain:
    # Precond:
    #   attributes is an integer representing the number of attributes in the
    #       domain.
    #   value is a list of integers. Each entry represent the number of unique
    #       values the cooresponding attribute can take.
    #   The length of values must equal or exceed that of the value of
    #       attributes. In the case of an excess all remaining values are
    #       ignored.
    #
    # Postcond:
    #   Builds a new Domain object with the specified parameters.
    def __init__(self, attributes, value):
        self.attributes = attributes
        self.value = value[0:self.attributes]
        self.attributes = len(self.value)

    # Precond:
    #   alt is a valid Alternative object for this domain, or None.
    #
    # Postcond:
    #   Iterator that yeilds each possible alternative in numerical order
    #   (least significant attribute first). If alt is specified then the
    #   iteration starts from that alternative.
    def each(self, alt=None):
        if alt is None:
            alt = Alternative([1 for i in range(self.attributes)])
        while not self.is_highest(alt):
            yield alt
            alt = self.next_alternative(alt)
        yield alt

    # Precond:
    #   None.
    #
    # Postcond:
    #   Iterator that yeilds each possible unique pair of alternatives in
    #   numerical order (least significant attribute first).
    def each_pair(self):
        for alt1 in self.each():
            for alt2 in self.each(self.next_alternative(alt1)):
                yield (alt1,alt2)


    # Precond:
    #   alt is a valid Alternative object.
    #
    # Postcond:
    #   Returns true if alt does not have the highest possible numerical value.
    def is_highest(self, alt):
        for i in range(self.attributes):
            if(alt.value(i) < self.value[i]):
                return False
        return True

    # Precond:
    #   alt is a valid Alternative object
    #
    # Postcond:
    #   Returns the next alternativein numerical order (least significant
    #   attribute first).
    def next_alternative(self, alt):
        for i in range(self.attributes):
            alt.set(i,alt.value(i)+1)
            if alt.value(i) >= self.value[i]:
                alt.set(i,1)
            else:
                return alt
        return alt
