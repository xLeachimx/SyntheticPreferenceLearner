import argparse
from lexicographic.lpm import LPM
from ranking.ranking_formula import RankingPrefFormula
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage


def main(args):
    agents = [args.agents[i] for i in range(len(args.agents))]
    return


def build_parser():
    parser = argparse.ArgumentParser(description="Automatically generate examples from randomly built synthetic agents.")
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('-f', dest='agent_folder', metavar='directory', type=str, nargs=1, help='Name of the agent output folder.', default=None)
    parser.add_argument('-n', dest='size', metavar='N', type=int, nargs=1, default=100, help="Number of examples to generate per agent.")
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser


if __name__=="__main__":
    main(build_parser().parse_args())
