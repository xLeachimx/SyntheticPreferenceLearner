# File: answer_set_optimization.py
# Author: Michael Huelsman
# Created On: 20 May 2020
# Purpose:
#   Contains a class which holds operations for dealing with ASO
#   preferences (assuming the generator program simply returns the set of all
#   alternatives in a given combinatorial domain).

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from untility.pref_logic import PrefFormula

class ASO:
    # Precond:
    #   domain is a valid Domain object.
    #
    # Postcond:
    #   Returns an empty ASO object.
    def __init__(self, domain):
        self.domain = domain
        self.ranks = []


    # Precond:
    #  domain is a valid Domain object.
    #   info is a dictionary containing all needed generating information.
    #       literals: The number of literals per clause.
    #       clauses: The number of clauses per formula.
    #       formulas: The number of formulas (satisfaction ranks) per rule.
    #       rules: The number of rules per rank.
    #       ranks: The number of ranks.
    #
    # Postcond:
    #   Returns a randomly generated ASO ovject of the specified type.
    @staticmethod
    def random(domain, info):
        result = ASO(domain)
        result.ranks = [None for i in range(info['ranks'])]
        # Generate each rank one at a time.
        for rank in range(info['ranks']):
            result.ranks[rank] = [None for i in range(info['rules'])]
            for rule in range(info['rules']):
                # Add a rule to the rank.
                result.ranks[rank][rule] = []
                for formula in range(info['formulas']):
                    # Add a random formula to the rule.
                    temp = PrefFormula.random(info['clauses'],info['literals'])
                    result.ranks[rank][rule].append(temp)
        return result

    # Precond:
    #   domain is a valid Domain object.
    #   info is a valid dictionary (data unused, but important for typing).
    #
    # Postcond:
    #   Returns a pill string describing the specified LPM
    @staticmethod
    def pill_label(domain, info):
        des = ['ASO',str(info['literals']),str(info['clauses']),str(info['formulas']),str(info['rules']),str(info['ranks']),str(domain.length()),str(domain.attr_length_largest())]
        return ';'.join(des)

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the string identifier of this class.
    @staticmethod
    def string_id():
        return "LPTree"
