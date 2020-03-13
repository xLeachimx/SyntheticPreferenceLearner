# File: example_set.py
# Author: Michael Huelsman
# Created On: 23 Oct 2019
# Purpose:
#   Defines an example set.
# Notes:
#   Updated on 25 Oct 2019
#   Updated on 31 Oct 2019

from random import shuffle
from torch.utils.data import Dataset

class ExampleSet(Dataset):
    # Precond:
    #   None.
    #
    # Postcond:
    #   Builds a new empty ExampleSet object
    def __init__(self):
        self.examples = {}
        self.examples['default'] = []

    # Precond:
    #   example is a valid Example object.
    #
    # Postcond:
    #   Files the example away based on the agent ID in the example.
    #   If there is no agent ID in the example the example is filed under
    #   the 'default' key.
    def add_example(self, example):
        agent = example.get_agent()
        if agent is None:
            self.examples['default'].append(example)
        else:
            if agent not in self.examples:
                self.examples[agent] = []
            self.examples[agent].append(example)

    # Precond:
    #   example is a valid Example object.
    #
    # Postcond:
    #   Files the example away based on the agent ID in the example.
    #   If there is no agent ID in the example the example is filed under
    #   the 'default' key.
    def add_example_list(self, examples):
        for example in examples:
            agent = example.get_agent()
            if agent is None:
                self.examples['default'].append(example)
            else:
                if agent not in self.examples:
                    self.examples[agent] = []
                self.examples[agent].append(example)

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns a single list with all examples.
    def example_list(self):
        result = []
        for id,lst in self.examples.items():
            result.extend(lst)
        return result

    # Precond:
    #   None.
    #
    # Postcond:
    #   Shuffles the lists, internally, per agent ID
    def shuffle(self):
        for id in self.examples:
            shuffle(self.examples[id])

    # Precond:
    #   None.
    #
    # Postcond:
    #   Returns the total number of examples in the set.
    def __len__(self):
        total = 0
        for id in self.examples:
            total += len(self.examples[id])
        return total

    # Precond:
    #   i is the index of the example ot retrieve.
    #
    # Postcond:
    #   Returns the ith example
    def __getitem__(self,i):
        for id in self.examples:
            if i >= len(self.examples[id]):
                i -= len(self.examples[id])
            else:
                return self.examples[id][i]
        return None

    # Precond:
    #   n is an integer representing the number of folds.
    #
    # Postcond:
    #   Returns an iterator which returns test and validation set.
    def crossvalidation(self,n):
        train = None
        validation = None
        self.shuffle()
        for i in range(n):
            temp_train = []
            temp_valid = []
            count = 0
            # Before the validation slice
            for id in self.examples:
                count = len(self.examples[id])/n
                temp_train.extend(self.examples[0:count*i])
            # The validation slice
            for id in self.examples:
                count = len(self.examples[id])/n
                temp_valid.extend(self.examples[count*i:count+(count*i)])
            # After the validation slice
            for id in self.examples:
                count = len(self.examples[id])/n
                temp_train.extend(self.examples[count+(count*i):])
            train = ExampleSet()
            validation = ExampleSet()
            train.add_example_list(temp_train)
            validation.add_example_list(temp_valid)
            yield (train,validation)
            del temp_train
            del temp_valid

    # TODO: Parsing methods
