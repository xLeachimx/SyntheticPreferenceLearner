import argparse
import os
from examples.agent import Agent
from examples.example_set import ExampleSet
from utility.configuration_parser import AgentHolder, parse_configuration
from lexicographic.lpm import LPM
from ranking.ranking_formula import RankingPrefFormula
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage


def main(args):
    print(args)
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage]
    agents = []
    # Build agents.
    for agent in config[1]:
        agents.append(make_agent(agent,agent_types,config[0]))
    # Write agents, if applicable
    if args.agent_folder is not None:
        if not os.path.isdir(args.agent_folder[0]):
            os.mkdir(args.agent_folder[0])
        for agent in agents:
            a_file = args.agent_folder[0] + "/agent"+str(agent[0].id)+".pref"
            with open(a_file, 'w') as fout:
                fout.write(str(agent[0].model))
    # Build example set.
    ex_set = ExampleSet()
    for agent in agents:
        temp_set = build_example_set(agent[0],agent[1],config[0])
        ex_set.add_example_list(temp_set.example_list())
    # Write example set to file.
    with open(args.output[0],'w') as fout:
        fout.write(str(config[0])+"\n")
        for example in ex_set.example_list():
            fout.write(str(example)+"\n")
    return




# Precond:
#   agent is a valid AgentHolder object.
#   types is the array of agent type classes.
#   domain is the domain of the agents.
#
# Postcond:
#   Returns tuple of:
#       1) A valid random Agent object of the type specfied by agent.
#       2) The number of examples to create.
def make_agent(agent, types, domain):
    for type in types:
        if agent.type.lower() == type.string_id().lower():
            return (Agent(type.random(domain,agent.info),domain),agent.size)
    return (None, 0)

# Precond:
#   agent is a valid Agent object.
#   size is the number of examples in the example set.
#   domain is the domain of the agent.
#
# Postcond:
#   Returns the example set for the agent.
def build_example_set(agent, size, domain):
    result = ExampleSet()
    pairs = domain.random_pair_set(size)
    for pair in pairs:
        result.add_example(agent.build_example(pair[0],pair[1]))
    return result

def build_parser():
    parser = argparse.ArgumentParser(description="Automatically generate examples from randomly built synthetic agents.")
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('-f', dest='agent_folder', metavar='directory', type=str, nargs=1, help='Name of the agent output folder.', default=None)
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser


if __name__=="__main__":
    main(build_parser().parse_args())
