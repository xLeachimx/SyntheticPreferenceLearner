import argparse
import os
import torch
import gc
from time import time
from examples.agent import Agent
from examples.example_set import ExampleSet
from examples.relation import Relation
from utility.configuration_parser import AgentHolder, parse_configuration
from lexicographic.lpm import LPM
from ranking.ranking_formula import RankingPrefFormula
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage
from conditional.cpnet import CPnet
from neural.neural_preferences import train_neural_preferences, prepare_example


def main(args):
    print(args)
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet]
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

def memReport():
    lst = []
    for obj in gc.get_objects():
        lst.append(str(obj))
    return lst

def cross_ref(lst1,lst2):
    result = []
    for obj in lst1:
        if not obj in lst2:
            result.append(obj)
    return result

def single_run(args, holder, agent_types, config, layers, learn_device):
    training = 0.0
    validation = 0.0
    agent = make_agent(holder,agent_types,config[0])
    ex_set = build_example_set(agent[0],agent[1],config[0])
    del agent
    proportion = ex_proport(ex_set)
    proportion = list(map(lambda x: str(x),proportion))
    proportion = ';'.join(proportion)
    for train, valid in ex_set.crossvalidation(5):
        # train.to_tensors(learn_device)
        # valid.to_tensors(learn_device)
        start = time()
        learner = train_neural_preferences(train,layers,1000,config[0],learn_device)
        # learner.to(eval_device)
        learner.eval()
        training = evaluate_cuda(train,learner,learn_device)
        validation = evaluate_cuda(valid,learner,learn_device)
        print(time()-start)
        # pills.append('(' + str(training) + ';' + str(validation) + ')')
        temp = ';'.join([str(training),str(validation),proportion])
        with open(args.output[0],'a') as fout:
            fout.write(',(' + temp + ')')
        torch.cuda.empty_cache()
        del temp
        del learner
        del train
        del valid
    del ex_set
    del proportion

def main_learn_nn(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet]
    agents = []
    learn_device = None
    # with open(args.output[0],'w') as fout:
    #     fout.write('')
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    layers = []
    for holder in config[1]:
        single_run(args, holder, agent_types, config, layers, learn_device)

# main for learning lpms
def main_learn_lpm(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet]
    for holder in config[1]:
        agent = make_agent(holder,agent_types,config[0])
        ex_set = build_example_set(agent[0],agent[1],config[0])
        proportion = ex_proport(ex_set)
        proportion = list(map(lambda x: str(x),proportion))
        proportion = ';'.join(proportion)
        for train, valid in ex_set.crossvalidation(5):
            start = time()
            learner = LPM.learn_greedy(train,config[0])
            print(time()-start)
            training = evaluate_rep(train,learner)
            validation = evaluate_rep(valid,learner)
            temp = ';'.join([str(training),str(validation),proportion])
            with open(args.output[0],'a') as fout:
                fout.write(',(' + temp + ')')


# Precond:
#   ex_set is a valid ExampleSet object
#
# Postcond:
#   Returns a list which provides a break down of the proportion of each
#   relation.
def ex_proport(ex_set):
    result = [0 for i in range(6)]
    for ex in ex_set.example_list():
        result[ex.get_relation().value+2] += 1
    for i in range(len(result)):
        result[i] = result[i]/float(len(ex_set))
    return result


# Precond:
#   ex_set is the example set to evaluate.
#   learner is the learner to evaluate.
#
# Postcond:
#   Returns the proportion of examples in the ex_set.
def evaluate_rep(ex_set, learner):
    count = 0
    for example in ex_set.example_list():
        alts = example.get_alts()
        if learner.compare(alts[0],alts[1]) == example.get_relation():
            count += 1
    return count/float(len(ex_set))

# Precond:
#   ex_set is the example set to evaluate.
#   learner is the learner to evaluate.
#
# Postcond:
#   Returns the proportion of examples in the ex_set.
def evaluate(ex_set, learner):
    count = 0
    for example in ex_set.example_list():
        inp,_ = prepare_example(example)
        label = Relation.parse_label(learner.forward_squash(inp))
        if label == example.get_relation():
            count += 1
        del inp
        del label
    return count/float(len(ex_set))

# Precond:
#   ex_set is the example set to evaluate.
#   learner is the learner to evaluate.
#   device is the device to run the tests on.
#
# Postcond:
#   Returns the proportion of examples in the ex_set.
def evaluate_cuda(ex_set, learner, device=None):
    count = 0
    for i in range(len(ex_set)):
        inp,expect = ex_set[i]
        if device is not None:
            inp = inp.to(device)
        label = learner.forward_squash(inp)#.to(torch.device('cpu'))
        label = Relation.parse_label(label)
        if label.value == expect-2:
            count += 1
        del inp
        del expect
        del label
    return count/float(len(ex_set))



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
    del pairs
    return result

def build_parser():
    parser = argparse.ArgumentParser(description="Automatically generate examples from randomly built synthetic agents.")
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser



if __name__=="__main__":
    # main_learn_nn(build_parser().parse_args())
    main_learn_lpm(build_parser().parse_args())
