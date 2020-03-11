# File: configuration_parser.py
# Author: Michael Huelsman
# Created On: 10 March 2020
# Purpose:
#   Handles parsing and storage of configuration files for the generation
#   of synthetic preferences.

from ..examples.domain import Domain

# Holds data for a single agent.
class AgentHolder:
    # Precond:
    #   type is a string identifying the type of agent.
    #   size is the number of examples the agent will produce.
    #   info is a dictionary containing additional information.
    #
    # Postcond:
    #   Builds a new AgentHolder object.
    def __init__(self, type, size, info):
        self.type = type
        self.size = size
        self.info = info


# Precond:
#   filename is the name of a valid configuration file.
#
# Postcond:
#   Returns a tuple containing, in order, these items:
#    1) Domain object.
#    2) List of AgentHolder objects.
#    3) True/False Flag indicating success/failure.
#    4) Error Message
def parse_configuration(filename):
    domain = None
    agents = []
    line = ''
    contents = []
    with open(filename,'r') as fin:
        line = fin.readline()
        while(line != ''):
            line = line.strip().lower()
            if line[0] != '#':
                if line[0] == 'd':
                    domain = Domain.parse(line)
                elif line == 'agent':
                    info = {}
                    type = None
                    size = 0
                    line = fin.readline().strip().lower()
                    while(line != 'end'):
                        contents = line.split(':')
                        contents = list(map(lambda x: x.strip().lower(), contents))
                        if contents
                        line = fin.readline().strip().lower()
            line = fin.readline()



    return (None,[], False, "Default Failure")
