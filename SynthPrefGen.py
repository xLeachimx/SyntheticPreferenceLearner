import argparse
from examples.agent import Agent
from utility.configuration_parser import AgentHolder, parse_configuration
from lexicographic.lpm import LPM
from ranking.ranking_formula import RankingPrefFormula
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage


def main(args):
    config = parse_configuration(args.config)
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage]
    agents = []
    for agent in config[1]:
        agents.append(make_agent(agent,agent_types,config[0]))
    return

def make_agent(agent, types, domain):
    for type in types:
        if agent.type.lower() == type.string_id().lower():
            return (Agent(type.random(domain,agent.info),domain),agent.size)
    return (None, 0)

def build_parser():
    parser = argparse.ArgumentParser(description="Automatically generate examples from randomly built synthetic agents.")
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('-f', dest='agent_folder', metavar='directory', type=str, nargs=1, help='Name of the agent output folder.', default=None)
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser


if __name__=="__main__":
    main(build_parser().parse_args())
