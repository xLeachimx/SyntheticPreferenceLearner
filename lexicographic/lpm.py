# File: lpm.py
# Author: Michael Huelsman
# Created On: 10 Dec 2019
# Purpose:
#   Specifies a class for handling Lexicographic Preference Models.

import random
from ../examples/relation import Relation

class LPM:
    # Precond:
    #   domain is a valid Domain object.
    #
    # Postcond:
    #   Produces an new empty LPM for the given domain.
    def __init__(self,domain):
        self.domain = domain
        self.importance = []
        self.orders = []


    # Precond:
    #   None.
    #
    # Postcond:
    #   Creates a randomly built LPM.
    def random(self):
        self.importance = [i for i in range(self.domain.length())]
        random.shuffle(self.importance)
        self.orders = [[] for i in range(self.domain.length())]
        for i in range(self.domain.length()):
            self.orders[i] = [(j+1) for j in range(self.domain.attr_length(i))]
            random.shuffle(self.orders[i])

    # Precond:
    #   None.
    #
    # Postcond:
    #   Creates a randomly built partial LPM.
    def random_partial(self):
        self.importance = [i for i in range(self.domain.length())]
        random.shuffle(self.importance)
        partial_point = random.randint(1,self.domain.length())
        self.importance = self.importance[:partial_point]
        self.orders = [[] for i in range(self.domain.length())]
        for i in range(self.domain.length()):
            self.orders[i] = [(j+1) for j in range(self.domain.attr_length(i))]
            random.shuffle(self.orders[i])


    # Precond:
    #  alt1 is a valid Alternative object.
    #  alt2 is a valid Alternative object.
    #
    # Postcond:
    #   Compares the two Altenratives and returns a Relation object.
    #   If the comparison cannot be completed returns None.
    def compare(self, alt1, alt2):
        if (not alt1.matches(self.domain)) or not (alt2.matches(self.domain)):
            return None
        for attr in self.importance:
            v1 = alt1.value(attr)
            v2 = alt2.value(attr)
            if v1 != v2:
                if self.orders[attr].index(v1) < self.orders[attr].index(v2):
                    return Relation.strict_preference()
                else:
                    return Relation.strict_dispreference()
        return Relation.equal()


    # Precond:
    #   lines is a set of strings which specifies an LPM.
    #   domain is the domain of the LPM described by the lines.
    #
    # Postcond:
    #   Returns the LPM specified by the given lines.
    @staticmethod
    def parse(lines, domain):
        result = LPM()
        result.domain = domain
        result.orders = [[] for i in range(domain.length())]
        convertList = lambda x: list(map(lambda y: int(y), x))
        for line in lines:
            line = line.strip()
            if line[0] == 'I':
                line = line.split(' ')[1:]
                result.importance = convertList(line)
            elif line[0] == 'P':
                line = line.split(' ')[1:]
                order = int(line[0])
                result.orders[order] = convertList(line[1:])
        return result

    # Precond:
    #   None.
    # Postcond:
    #   Builds a string representation of the LPM.
    def __str__(self):
        result = str(domain)+"\n"
        result += 'I '
        result += ' '.join(list(map(lambda x: str(x), self.importance))) + "\n"
        for i in range(len(self.orders)):
            result += 'P ' + str(i) + ' '.join(list(map(lambda x: str(x), self.orders[i]))) + "\n"
        return result
