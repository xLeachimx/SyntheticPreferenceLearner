# File: penalty_logic.py
# Author: Michael Huelsman
# Created On: 13 Dec 2019
# Purpose:
#   Provides a class for dealing with penalty logic preference representations.

import random
import math
from ..utility.pref_logic import PrefFormula
from ..examples.relation import Relation

class PenaltyLogic:
    # Precond:
    #   domain is a valid Domain object.
    #
    # Postcond:
    #   Builds a new empty penalty logic preference representation.
    def __init__(self, domain):
        self.domain = domain
        self.formulas = []
        self.weights = []

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the number of formulas.
    def length(self):
        return len(self.formulas)

    # Precond:
    #   alt is a valud Alternative object.
    #
    # Postcond:
    #   Returns the numerical valus of the given alternative.
    #   Evluation done on formulas is CNF.
    #   Defaults to -1 (impossible value)
    def eval_CNF(self, alt):
        if not alt.matches(domain):
            return -1.0
        total = 0.0
        for i in range(len(self.formulas)):
            if not self.formulas[i].eval_CNF(alt):
                total += self.weights[i]
        return total

    # Precond:
    #   alt is a valud Alternative object.
    #
    # Postcond:
    #   Returns the numerical valus of the given alternative.
    #   Evluation done on formulas is DNF.
    #   Defaults to -1 (impossible value)
    def eval_DNF(self, alt):
        if not alt.matches(domain):
            return -1.0
        total = 0.0
        for i in range(len(self.formulas)):
            if not self.formulas[i].eval_DNF(alt):
                total += self.weights[i]
        return total

    # Precond:
    #   alt1 is a valid Alternative object.
    #   alt2 is a valud Alternative object.
    #   dnf is a boolean which is true when DNF evaluation is used.
    #
    # Postcond:
    #   Returns the relation between alt1 and alt2.
    def compare(self, alt1, alt2, dnf=True):
        val1 = val2 = 0.0
        if dnf:
            val1 = self.eval_DNF(alt1)
            val2 = self.eval_DNF(alt2)
        else:
            val1 = self.eval_CNF(alt1)
            val2 = self.eval_CNF(alt2)
        if val1 > val2:
            return Relation.strict_dispreference()
        elif val1 < val2:
            return Relation.strict_preference()
        return Relation.equal()


    # Precond:
    #   domain is a valid Domain object.
    #   info is a valid dictionary with the following keys:
    #       size is the number of formulas to generate.
    #       clauses is the number of clauses per formula.
    #       literals is the number of literals per clause.
    #
    # Postcond:
    #   Returns a randomly generated PenaltyLogic object.
    @staticmethod
    def random(domain, info):
        result = PenaltyLogic(domain)
        generate = lambda : PrefFormula.random(info['clauses'],info['literals'],domain)
        result.formulas = [generate() for i in range(info['formulas'])]
        result.weights = [random.random for i in range(info['formulas'])]
        total = 0.0
        for weight in result.weights:
            total += weight*weight
        total = math.sqrt(total)
        result.weights = list(map(lambda x: x/total, result.weights))
        return result

    # Precond:
    #   Lines is a list of strings which describe a PenaltyLogic object.
    #   domain is a valid Domain object.
    #
    # Postcond:
    #   Returns the parsed PenaltyLogic object.
    @staticmethod
    def parse(lines,domain):
        result = PenaltyLogic(domain)
        for line in lines:
            line = line.strip()
            if line[0] == 'P':
                line = line.split(' ')
                result.weights.append(float(line[1]))
                line = ' '.join(line[2:])
                result.formulas.append(PrefFormula.parse(line,domain))
        return result

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the string identifier of this class.
    @staticmethod
    def string_id():
        return "PenaltyLogic"

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the string representation of tht PenaltyLogic object.
    def __str__(self):
        result = str(self.domain) + "\n"
        for i in range(len(self.formulas)):
            result += ' ',join(['P',str(self.weights[i]),str(self.formulas[i])])
            result += "\n"
        return result
