import argparse
import time
import torch
from os import system
from utility.configuration_parser import AgentHolder, parse_configuration
from lexicographic.lpm import LPM
from lexicographic.lp_tree import LPTree
from ranking.ranking_formula import RankingPrefFormula
from ranking.answer_set_optimization import ASO
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage
from conditional.cpnet import CPnet
from conditional.clpm import CLPM
from conditional.cpnet import CPnet
import multiprocessing as mp

def main(args, mp_n):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    config = parse_configuration(args.config[0])
    with open(args.output[0],'w') as fout:
        fout.write('')
    for holder in config[1]:
        with open('temp_agent.config','w') as fout:
            fout.write(str(config[0])+"\n")
            fout.write(str(holder))
        call = "python3 SynthPrefGen.py -l " + str(args.layers[0]) + ' '
        prob = list(map(lambda x: str(x),args.problem))
        call += '-p ' + ' '.join(prob) + ' '
        if len(args.learn_conf) == 1:
            call += '-i ' + args.learn_conf[0] + ' '
        call += "-o " + args.output[0] + " "
        call += "temp_agent.config >> timing.dat"
        runs = 25
        print(call)
        with open(args.output[0],'a') as fout:
            fout.write('(' + pill_label(agent_types,holder,config[0]) + ';' + str(holder.size) +  ')')
        pool = mp.Pool(mp_n)
        pool.map(sys_call_wait,[call for i in range(runs)])
        with open(args.output[0],'a') as fout:
            fout.write("\n")
    return 0

def main_multi(args, mp_n):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    config = parse_configuration(args.config[0])
    with open(args.output[0],'w') as fout:
        fout.write('')
    call = "python3 SynthPrefGen.py -l " + str(args.layers[0]) + ' '
    prob = list(map(lambda x: str(x),args.problem))
    call += '-p ' + ' '.join(prob) + ' '
    if len(args.learn_conf) == 1:
        call += '-i ' + args.learn_conf[0] + ' '
    call += "-o " + args.output[0] + " "
    call += args.config[0] + " >> timing.dat"
    runs = 25
    label = ''
    for holder in config[1]:
        label += pill_label(agent_types,holder,config[0])
    label += ';' + str(config[1][0].size)
    with open(args.output[0],'a') as fout:
        fout.write('(' + label +  ')')
    pool = mp.Pool(mp_n)
    pool.map(sys_call_wait,[call for i in range(runs)])

    with open(args.output[0],'a') as fout:
        fout.write("\n")
    return 0

def sys_call_wait(call):
    system(call)
    time.sleep(5)


def main_neighbor(args, mp_n):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    config = parse_configuration(args.config[0])
    with open(args.output[0],'w') as fout:
        fout.write('')
    call = "python3 SynthPrefGen.py -l " + str(args.layers[0]) + ' '
    prob = list(map(lambda x: str(x),args.problem))
    call += '-p ' + ' '.join(prob) + ' '
    if len(args.learn_conf) == 1:
        call += '-i ' + args.learn_conf[0] + ' '
    call += "-o " + args.output[0] + " "
    call += args.config[0] + " >> timing.dat"
    runs = 25
    label = ''
    for holder in config[1]:
        label += pill_label(agent_types,holder,config[0])
    label += ';' + str(config[1][0].size)
    with open(args.output[0],'a') as fout:
        fout.write('(' + label +  ')')
    start = time.time()
    pool = mp.Pool(mp_n)
    pool.map(sys_call_wait,[call for i in range(runs)])
    with open(args.output[0],'a') as fout:
        fout.write("\n")
    print("Full Time:",time.time()-start)
    return 0

def main_portfolio(args,mp_n):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    config = parse_configuration(args.config[0])
    with open(args.output[0],'w') as fout:
        fout.write('')
    call = "python3 SynthPrefGen.py -l " + str(args.layers[0]) + ' '
    prob = list(map(lambda x: str(x),args.problem))
    call += '-p ' + ' '.join(prob) + ' '
    if len(args.learn_conf) == 1:
        call += '-i ' + args.learn_conf[0] + ' '
    call += "-o " + args.output[0] + " "
    # call += args.config[0] + " >> timing.dat"
    call += args.config[0]
    runs = 25
    label = []
    for holder in config[1]:
        label.append(pill_label(agent_types,holder,config[0]))
    label.append(str(config[1][0].size))
    label = '|'.join(label)
    with open(args.output[0],'a') as fout:
        fout.write('(' + label +  ')')
    pool = mp.Pool(mp_n)
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
    parser.add_argument('-p', dest='problem', metavar='n', type=int, nargs=2, default=[1,1], help='Specified which problem/subproblem to run.')
    parser.add_argument('-l', dest='layers', metavar='n', type=int, nargs=1, default=[3], help='The number of neural net layers')
    parser.add_argument('-i', dest='learn_conf', metavar='filename', type=str, nargs=1, help='Name of the learner configuration file.', default='a.exs')
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser


if __name__=="__main__":
    if torch.cuda.is_available():
        print("Cuda Online")
    processor_pool = 4
    args = build_parser().parse_args()
    print(args.problem)
    if args.problem[0] == 1 or (args.problem[0] == 4 and (args.problem[1] == 4 or args.problem[1] == 5 or args.problem[1] == 6 or args.problem[1] == 7)):
        # Number of process to use when learning NNs.
        processor_pool = 2
    if args.problem[0] == 4:
        if args.problem[1] == 4 or args.problem[1] == 5 or args.problem[1] == 6 or args.problem[1] == 7:
            main_portfolio(args,processor_pool)
        else:
            main_neighbor(args,processor_pool)
    else:
        if args.problem[1] == 2 or args.problem[1] == 3:
            main_multi(args,processor_pool)
        else:
            main(args,processor_pool)
