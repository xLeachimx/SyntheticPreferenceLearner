import argparse
from os import system
from utility.configuration_parser import AgentHolder, parse_configuration
from lexicographic.lpm import LPM
from ranking.ranking_formula import RankingPrefFormula
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage
from conditional.cpnet import CPnet

def main(args):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet]
    config = parse_configuration(args.config[0])
    with open(args.output[0],'w') as fout:
        fout.write('')
    for holder in config[1]:
        with open('temp_agent.config','w') as fout:
            fout.write(str(config[0])+"\n")
            fout.write(str(holder))
        call = "python3 SynthPrefGen.py -o " + args.output[0] + " "
        call += "temp_agent.config >> timing.dat"
        runs = 25
        with open(args.output[0],'a') as fout:
            fout.write('(' + pill_label(agent_types,holder,config[0]) + ';' + str(holder.size) +  ')')
        for i in range(runs):
            print(i)
            system(call)
        with open(args.output[0],'a') as fout:
            fout.write("\n")
    return 0

def pill_label(types, holder, domain):
    for type in types:
        if holder.type.lower() == type.string_id().lower():
            return type.pill_label(domain,holder.info)
    return 'None'

def build_parser():
    parser = argparse.ArgumentParser(description="Automatically generate examples from randomly built synthetic agents.")
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser


if __name__=="__main__":
    main(build_parser().parse_args())
