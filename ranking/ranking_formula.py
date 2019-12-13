# File: ranking_formula.py
# Author: Michael Huelsman
# Created On: 12 Dec 2019
# Purpose:
#   Define a ranking preference formula.

from ..languages.pref_logic import PrefFormula
from ..examples.relation import Relation

class RankingPrefFormula:
    # Precond:
    #   domain is a valid Domain object.
    #
    # Postcond:
    #   Builds a new empty RankingPrefFormula object.
    def __init__(self, domain):
        self.domain = domain
        self.ranks = []

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the number of ranks in the RankingPrefFormula object.
    def length(self):
        return len(self.ranks)

    # Precond:
    #   alt is a valid Alternative object.
    #
    # Postcond:
    #   Returns the first rank where the alternative matches the DNF formula.
    #   Defaults to returning # of ranks + 1.
    def eval_DNF(self, alt):
        if not alt.matches(self.domain):
            return len(self.ranks)+1
        for i in range(len(self.ranks)):
            if self.ranks[i].eval_DNF(alt):
                return i+1
        return len(self.ranks)+1

    # Precond:
    #   alt is a valid Alternative object.
    #
    # Postcond:
    #   Returns the first rank where the alternative matches the CNF formula.
    #   Defaults to returning # of ranks + 1.
    def eval_DNF(self, alt):
        if not alt.matches(self.domain):
            return len(self.ranks)+1
        for i in range(len(self.ranks)):
            if self.ranks[i].eval_CNF(alt):
                return i+1
        return len(self.ranks)+1

    # Precond:
    #   alt1 is a valid Alternative object.
    #   alt2 is a valud Alternative object.
    #   dnf is a boolean which is true when DNF evaluation is used.
    # 
    # Postcond:
    #   Returns the relation between alt1 and alt2.
    def compare(self, alt1, alt2, dnf=True):
        rank1 = rank2 = 0
        if dnf:
            rank1 = self.eval_DNF(alt1)
            rank2 = self.eval_DNF(alt2)
        else:
            rank1 = self.eval_CNF(alt1)
            rank2 = self.eval_CNF(alt2)
        if rank1 < rank2:
            return Relation.strict_dispreference()
        elif rank1 > rank2:
            return Relation.strict_preference()
        return Relation.equal()

    # Precond:
    #   clauses is the number of clauses per rank.
    #   literals is the number of literals per clause.
    #   ranks is the number of ranks to generate.
    #   domain is a valid Domain object.
    #
    # Postcond:
    #   Returns a random RankingPrefFormula object.
    @staticmethod
    def random(clauses, literals, ranks, domain):
        result = RankingPrefFormula(domain)
        for i in range(ranks):
            result.ranks.append(PrefFormula.random(clauses,literals))
        return result

    # Precond:
    #   lines is a list strings representing rank formulas (in order).
    #   domain is a valid Domain object.
    #
    # Postcond:
    #   Returns a RankingPrefFormula object.
    @staticmethod
    def parse(lines, domain):
        result = RankingPrefFormula(domain)
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            if lines[i][0] == 'P':
                lines[i] = lines[i][1:]
                result.ranks.append(PrefFormula.parse(lines[i],domain))

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns a string representation of the ranking formula.
    def __str__(self):
        result = str(self.domain) + "\n"
        for i in range(len(self.ranks)):
            result += ' '.join(['P', str(self.ranks[i]), "\n"])
