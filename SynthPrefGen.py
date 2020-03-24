import argparse
import os
import torch
from examples.agent import Agent
from examples.example_set import ExampleSet
from examples.relation import Relation
from utility.configuration_parser import AgentHolder, parse_configuration
from lexicographic.lpm import LPM
from ranking.ranking_formula import RankingPrefFormula
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage
from neural.neural_preferences import train_neural_preferences, prepare_example


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

def main_learn(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage]
    agents = []
    learn_device = None
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    # Build agents and example sets.
    for holder in config[1]:
        # 50 trials
        training = 0.0
        validation = 0.0
        runs = 1
        layers = [256,256,256]
        pills = []
        pills.append('(' + holder.type + ';' + str(holder.size) +  ')')
        for i in range(runs):
            print(i)
            agent = make_agent(holder,agent_types,config[0])
            ex_set = build_example_set(agent[0],agent[1],config[0])
            print(agent[0])
            ex_set.to_tensors(learn_device)
            proportion = ex_proport(ex_set)
            proportion = list(map(lambda x: str(x),proportion))
            proportion = ';'.join(proportion)
            for train, valid in ex_set.crossvalidation(5):
                # train.to_tensors(learn_device)
                # valid.to_tensors(learn_device)
                learner = train_neural_preferences(train,layers,1000,config[0])
                training = evaluate(train,learner)
                validation = evaluate(valid,learner)
                # pills.append('(' + str(training) + ';' + str(validation) + ')')
                temp = ';'.join([str(training),str(validation),proportion])
                pills.append('(' + temp + ')')
            del ex_set
            del agent
        with open(args.output[0],'a') as fout:
            fout.write(','.join(pills) + "\n")
        del pills

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
def evaluate(ex_set, learner):
    count = 0
    for example in ex_set.example_list():
        inp,_ = prepare_example(example)
        label = Relation.parse_label(learner.forward(inp))
        if label == example.get_relation():
            count += 1
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
    return result

def build_parser():
    parser = argparse.ArgumentParser(description="Automatically generate examples from randomly built synthetic agents.")
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('-f', dest='agent_folder', metavar='directory', type=str, nargs=1, help='Name of the agent output folder.', default=None)
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser


if __name__=="__main__":
    main_learn(build_parser().parse_args())
