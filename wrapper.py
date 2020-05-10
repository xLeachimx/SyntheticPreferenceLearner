import argparse
import time
from os import system
from utility.configuration_parser import AgentHolder, parse_configuration
from lexicographic.lpm import LPM
from ranking.ranking_formula import RankingPrefFormula
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage
from conditional.cpnet import CPnet
import multiprocessing as mp

def main(args):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet]
    config = parse_configuration(args.config[0])
    with open(args.output[0],'w') as fout:
        fout.write('')
    for holder in config[1]:
        with open('temp_agent.config','w') as fout:
            fout.write(str(config[0])+"\n")
            fout.write(str(holder))
        call = "python3 SynthPrefGen.py -l " + str(args.layers) + ' '
        if len(args.learn_conf) == 1:
            call += '-i ' + args.learn_conf[0] + ' '
        call += "-o " + args.output[0] + " "
        call += "temp_agent.config >> timing.dat"
        runs = 25
        with open(args.output[0],'a') as fout:
            fout.write('(' + pill_label(agent_types,holder,config[0]) + ';' + str(holder.size) +  ')')
        pool = mp.Pool()
        pool.map(sys_call_wait,[call for i in range(runs)])
        with open(args.output[0],'a') as fout:
            fout.write("\n")
    return 0

def main_multi(args):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet]
    config = parse_configuration(args.config[0])
    with open(args.output[0],'w') as fout:
        fout.write('')
    call = "python3 SynthPrefGen.py -o " + args.output[0] + " "
    if len(args.learn_conf) == 1:
        call += '-i ' + args.learn_conf[0] + ' '
    call += args.config[0] + " >> timing.dat"
    runs = 25
    label = ''
    for holder in config[1]:
        label += pill_label(agent_types,holder,config[0])
    label += ';' + str(config[1][0].size)
    with open(args.output[0],'a') as fout:
        fout.write('(' + label +  ')')
    pool = mp.Pool(2)
    pool.map(sys_call_wait,[call for i in range(runs)])

    with open(args.output[0],'a') as fout:
        fout.write("\n")
    return 0

def sys_call_wait(call):
    system(call)
    time.sleep(5)


def main_neighbor(args):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet]
    config = parse_configuration(args.config[0])
    with open(args.output[0],'w') as fout:
        fout.write('')
    call = "python3 SynthPrefGen.py -o " + args.output[0] + " "
    if len(args.learn_conf) == 1:
        call += '-i ' + args.learn_conf[0] + ' '
    call += args.config[0] + " >> timing.dat"
    runs = 25
    label = ''
    for holder in config[1]:
        label += pill_label(agent_types,holder,config[0])
    label += ';' + str(config[1][0].size)
    with open(args.output[0],'a') as fout:
        fout.write('(' + label +  ')')
    pool = mp.Pool(4)
    pool.map(sys_call_wait,[call for i in range(runs)])
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
    parser.add_argument('-l', dest='layers', metavar='n', type=int, nargs=1, default=3, help='The number of neural net layers')
    parser.add_argument('-i', dest='learn_conf', metavar='filename', type=str, nargs=1, help='Name of the learner configuration file.', default='a.exs')
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser


if __name__=="__main__":
    main(build_parser().parse_args())
    # main_multi(build_parser().parse_args())
    # main_neighbor(build_parser().parse_args())