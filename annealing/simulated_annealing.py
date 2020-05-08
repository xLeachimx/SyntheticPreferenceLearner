# File: simulated_annealing.py
# Author: Michael Huelsman
# Created On: 8 April 2020
# Purpose:
#   Houses a general simulated annealing algorithm for learning preferences.

from random import random
from math import exp

# Precond:
#   learner is a preference represntation object implementing the following methods:
#       1) neighbors(self) -> Iterates through all neighbors of the learner.
#       2) random_neighbor(self) -> Returns a random neighbor of the learner.
#       3) compare(self, alt1, alt2) -> Returns the relation between alt1 and alt2.
#   ex_set is a valid ExampleSet object.
#
# Postcond:
#   Returns a modified learner which has been processed through an implementation
#   of simulated annealing and then a round of hill climbing.
def learn_SA(learner, ex_set):
    current_eval = evaluate_util(learner, ex_set)
    temp = 100
    cool = 0.001
    while temp > 10**(-6):
        neighbor = learner.random_neighbor()
        eval = evaluate_util(neighbor, ex_set)
        if eval > current_eval:
            current_eval = eval
            learner = neighbor
        else:
            delta = eval - current_eval
            prob = exp(delta/temp)
            if random() <= prob:
                learner = neighbor
                current_eval = eval
        temp = temp/(1.0+cool)
    return hillclimb(learner,ex_set)

# Precond:
#   learner is a preference represntation object implementing the following methods:
#       1) neighbors(self) -> Iterates through all neighbors of the learner.
#       2) compare(self, alt1, alt2) -> Returns the relation between alt1 and alt2.
#   ex_set is a valid ExampleSet object.
#
# Postcond:
#   Returns a modified learner which has been processed through an implementation
#   of hill climbing.
def hillclimb(learner, ex_set):
    best = learner
    best_eval = evaluate_util(learner, ex_set)
    improved = True
    while improved:
        improved = False
        for neighbor in learner.neighbors():
            eval = evaluate_util(neighbor, ex_set)
            if eval > best_eval:
                improved = True
                best = neighbor
                best_eval = eval
    return learner

# Precond:
#   learner is a preference represntation object implementing the following methods:
#       3) compare(self, alt1, alt2) -> Returns the relation between alt1 and alt2.
#   ex_set is a valid ExampleSet object.
#
# Postcond:
#   Returns the proportion of examples satisfied by the learner in the example set.
def evaluate_util(learner, ex_set):
    correct = 0
    for ex in ex_set.each():
        alts = ex.get_alts()
        if learner.compare(alts[0],alts[1]) == ex.get_relation():
            correct += 1
    return correct/float(len(ex_set))
