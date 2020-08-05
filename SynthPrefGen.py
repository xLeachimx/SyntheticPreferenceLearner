import argparse
import os
import torch
import gc
from time import time
from examples.agent import Agent
from examples.example_set import ExampleSet
from examples.relation import Relation
from utility.configuration_parser import AgentHolder, parse_configuration
from utility.neighbor_graph import NeighborGraph
from utility.preference_graph import PreferenceGraph
from lexicographic.lpm import LPM
from lexicographic.lp_tree import LPTree
from ranking.ranking_formula import RankingPrefFormula
from ranking.answer_set_optimization import ASO
from weighted.penalty_logic import PenaltyLogic
from weighted.weighted_average import WeightedAverage
from conditional.cpnet import CPnet
from conditional.clpm import CLPM
from neural.neural_preferences import train_neural_preferences, prepare_example
from annealing.simulated_annealing import learn_SA, learn_SA_mm
import annealing.simulated_annealing as SA


def main(args):
    print(args)
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
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

def single_run_full(args, holder, agent_types, config, layers, learn_device):
    training = 0.0
    validation = 0.0
    agent = make_agent(holder,agent_types,config[0])
    ex_set = build_example_set(agent[0],agent[1],config[0])
    for train, valid in ex_set.crossvalidation(5):
        # train.to_tensors(learn_device)
        # valid.to_tensors(learn_device)
        start = time()
        learner = train_neural_preferences(train,layers,1000,config[0],learn_device)
        # learner.to(eval_device)
        learner.eval()
        training = evaluate_cuda(train,learner,learn_device)
        validation = evaluate_cuda(valid,learner,learn_device)
        # full_validation = full_cuda_eval(config[0],learner,agent[0],learn_device)
        full_validation = -1
        p_graph = build_NN_pref_graph(config[0],learner,learn_device)
        print(time()-start)
        # pills.append('(' + str(training) + ';' + str(validation) + ')')
        temp = ';'.join([str(training),str(validation),str(full_validation),str(p_graph.cyclicity())])
        with open(args.output[0],'a') as fout:
            fout.write(',(' + temp + ')')
        torch.cuda.empty_cache()
        del p_graph
        del temp
        del learner
        del train
        del valid
    del agent
    del ex_set

def main_learn_nn(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    agents = []
    learn_device = None
    # with open(args.output[0],'w') as fout:
    #     fout.write('')
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    layers = [256,256,256]
    layer_cut = max(0,args.layers[0])
    layers = layers[:layer_cut]
    for holder in config[1]:
        single_run(args, holder, agent_types, config, layers, learn_device)

def main_learn_nn_full(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    agents = []
    learn_device = None
    # with open(args.output[0],'w') as fout:
    #     fout.write('')
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    layers = [256,256,256]
    layer_cut = max(0,args.layers[0])
    layers = layers[:layer_cut]
    for holder in config[1]:
        single_run_full(args, holder, agent_types, config, layers, learn_device)

# main for learning lpms
def main_learn_lpm(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
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

# main for learning lpms
def main_learn_joint_lpm(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    agents = []
    for holder in config[1]:
        agents.append(make_agent(holder,agent_types,config[0]))
    ex_set = build_example_set_multi(agents, config[0])
    for train, valid in ex_set.crossvalidation(5):
        start = time()
        learner = LPM.learn_greedy(train,config[0])
        print(time()-start)
        training = evaluate_multi(train,learner)
        validation = evaluate_multi(valid,learner)
        training = ';'.join(list(map(lambda x: str(x),training)))
        validation = ';'.join(list(map(lambda x: str(x),validation)))
        temp = ';'.join([training,validation])
        with open(args.output[0],'a') as fout:
            fout.write(',(' + temp + ')')

# main for learning lpms
def main_learn_joint_lpm_mm(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    agents = []
    for holder in config[1]:
        agents.append(make_agent(holder,agent_types,config[0]))
    ex_set = build_example_set_multi(agents, config[0])
    for train, valid in ex_set.crossvalidation(5):
        start = time()
        learner = LPM.learn_greedy_maximin(train,config[0])
        print(time()-start)
        training = evaluate_multi(train,learner)
        validation = evaluate_multi(valid,learner)
        training = ';'.join(list(map(lambda x: str(x),training)))
        validation = ';'.join(list(map(lambda x: str(x),validation)))
        temp = ';'.join([training,validation])
        with open(args.output[0],'a') as fout:
            fout.write(',(' + temp + ')')

def main_learn_lpm_full(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    for holder in config[1]:
        agent = make_agent(holder,agent_types,config[0])
        ex_set = build_example_set(agent[0],agent[1],config[0])
        for train, valid in ex_set.crossvalidation(5):
            start = time()
            learner = LPM.learn_greedy(train,config[0])
            print(time()-start)
            training = evaluate_rep(train,learner)
            validation = evaluate_rep(valid,learner)
            full_eval = evaluate_rep_full(agent[0], learner, config[0])
            temp = ';'.join([str(training),str(validation),str(full_eval)])
            with open(args.output[0],'a') as fout:
                fout.write(',(' + temp + ')')

def main_learn_joint_nn(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    info = {}
    agents = []
    learn_device = None
    # with open(args.output[0],'w') as fout:
    #     fout.write('')
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    layers = [256,256,256]
    layer_cut = max(0,args.layers[0])
    layers = layers[:layer_cut]
    for holder in config[1]:
        agents.append(make_agent(holder,agent_types,config[0]))
    ex_set = build_example_set_multi(agents, config[0])
    for train, valid in ex_set.crossvalidation(5):
        # train.to_tensors(learn_device)
        # valid.to_tensors(learn_device)
        start = time()
        learner = train_neural_preferences(train,layers,1000,config[0],learn_device)
        # learner.to(eval_device)
        learner.eval()
        training = evaluate_cuda_multi(train,learner,learn_device)
        validation = evaluate_cuda_multi(valid,learner,learn_device)
        print(time()-start)
        # pills.append('(' + str(training) + ';' + str(validation) + ')')
        training = ';'.join(list(map(lambda x: str(x),training)))
        validation = ';'.join(list(map(lambda x: str(x),validation)))
        temp = ';'.join([training,validation])
        with open(args.output[0],'a') as fout:
            fout.write(',(' + temp + ')')
        torch.cuda.empty_cache()
        del temp
        del learner
        del train
        del valid
    del ex_set

def main_learn_joint_SA(args):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    config = parse_configuration(args.config[0])
    info = {}
    l_class = None
    if len(args.learn_conf) == 1:
        l_config = parse_configuration(args.learn_conf[0])
        info = l_config[1][0].info
        for type in agent_types:
            if l_config[1][0].type.lower() == type.string_id().lower():
                l_class = type
    else:
        info['clauses'] = 1
        info['literals'] = 1
        info['ranks'] = 5
        l_class = RankingPrefFormula

    agents = []
    for holder in config[1]:
        agents.append(make_agent(holder,agent_types,config[0]))
    ex_set = build_example_set_multi(agents, config[0])
    for train, valid in ex_set.crossvalidation(5):
        start = time()
        learner = l_class.random(config[0],info)
        # learner = LPM.random(config[0], info)
        learner = learn_SA(learner, train)
        print(time()-start)
        training = evaluate_multi(train,learner)
        validation = evaluate_multi(valid,learner)
        training = ';'.join(list(map(lambda x: str(x),training)))
        validation = ';'.join(list(map(lambda x: str(x),validation)))
        temp = ';'.join([training,validation])
        with open(args.output[0],'a') as fout:
            fout.write(',(' + temp + ')')
        del temp
        del learner
        del train
        del valid
    del ex_set

def main_learn_joint_SA_mm(args):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    config = parse_configuration(args.config[0])
    info = {}
    l_class = None
    if len(args.learn_conf) == 1:
        l_config = parse_configuration(args.learn_conf[0])
        info = l_config[1][0].info
        for type in agent_types:
            if l_config[1][0].type.lower() == type.string_id().lower():
                l_class = type
    else:
        info['clauses'] = 1
        info['literals'] = 1
        info['ranks'] = 5
        l_class = RankingPrefFormula

    agents = []
    for holder in config[1]:
        agents.append(make_agent(holder,agent_types,config[0]))
    ex_set = build_example_set_multi(agents, config[0])
    for train, valid in ex_set.crossvalidation(5):
        start = time()
        learner = l_class.random(config[0],info)
        # learner = LPM.random(config[0], info)
        learner = learn_SA_mm(learner, train)
        print(time()-start)
        training = evaluate_multi(train,learner)
        validation = evaluate_multi(valid,learner)
        training = ';'.join(list(map(lambda x: str(x),training)))
        validation = ';'.join(list(map(lambda x: str(x),validation)))
        temp = ';'.join([training,validation])
        with open(args.output[0],'a') as fout:
            fout.write(',(' + temp + ')')
        del temp
        del learner
        del train
        del valid
    del ex_set

# main for learning using simulated annealing
def main_learn_SA(args):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    config = parse_configuration(args.config[0])
    info = {}
    l_class = None
    if len(args.learn_conf) == 1:
        l_config = parse_configuration(args.learn_conf[0])
        info = l_config[1][0].info
        for type in agent_types:
            if l_config[1][0].type.lower() == type.string_id().lower():
                l_class = type
    else:
        info['clauses'] = 1
        info['literals'] = 1
        info['ranks'] = 5
        l_class = RankingPrefFormula

    for holder in config[1]:
        agent = make_agent(holder,agent_types,config[0])
        ex_set = build_example_set(agent[0],agent[1],config[0])
        proportion = ex_proport(ex_set)
        proportion = list(map(lambda x: str(x),proportion))
        proportion = ';'.join(proportion)
        for train, valid in ex_set.crossvalidation(5):
            start = time()
            learner = l_class.random(config[0],info)
            # learner = RankingPrefFormula.random(config[0],info)
            # learner = LPM.random(config[0], info)
            learner = learn_SA(learner, train)
            print(time()-start)
            training = evaluate_rep(train,learner)
            validation = evaluate_rep(valid,learner)
            temp = ';'.join([str(training),str(validation),proportion])
            with open(args.output[0],'a') as fout:
                fout.write(',(' + temp + ')')

def main_learn_SA_full(args):
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    config = parse_configuration(args.config[0])
    info = {}
    l_class = None
    if len(args.learn_conf) == 1:
        l_config = parse_configuration(args.learn_conf[0])
        info = l_config[1][0].info
        for type in agent_types:
            if l_config[1][0].type.lower() == type.string_id().lower():
                l_class = type
    else:
        info['clauses'] = 1
        info['literals'] = 1
        info['ranks'] = 5
        l_class = RankingPrefFormula

    for holder in config[1]:
        agent = make_agent(holder,agent_types,config[0])
        ex_set = build_example_set(agent[0],agent[1],config[0])
        for train, valid in ex_set.crossvalidation(5):
            start = time()
            learner = l_class.random(config[0],info)
            # learner = RankingPrefFormula.random(config[0],info)
            # learner = LPM.random(config[0], info)
            learner = learn_SA(learner, train)
            print(time()-start)
            training = evaluate_rep(train,learner)
            validation = evaluate_rep(valid,learner)
            full_eval = evaluate_rep_full(agent[0], learner, config[0])
            temp = ';'.join([str(training),str(validation),str(full_eval)])
            with open(args.output[0],'a') as fout:
                fout.write(',(' + temp + ')')

def main_build_neighbor(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    info = {}
    info['clauses'] = 1
    info['literals'] = 1
    info['ranks'] = 3
    agents = []
    for holder in config[1]:
        agents.append(make_agent(holder,agent_types,config[0]))
    ex_set = build_example_set_multi(agents, config[0])
    n_graph = NeighborGraph()
    agents = list(map(lambda x: x[0], agents))
    start = time()
    for rpf in RankingPrefFormula.each(config[0],info):
        # eval = evaluate_rep_full_multi(agents, rpf, config[0])
        # eval = evaluate_rep_full_maximin(agents, rpf, config[0])
        # eval = evaluate_rep(ex_set, rpf)
        eval = SA.evaluate_maximin(rpf, ex_set)
        n_graph.add_node(rpf.node_str(), eval)
        for neighbor in rpf.neighbors():
            n_graph.add_arc(rpf.node_str(), neighbor.node_str())
    maxima = n_graph.local_maxima()
    average_maxima = 0.0
    min_maxima = None
    for node in maxima:
        average_maxima += n_graph.get(node)
        if min_maxima is None or n_graph.get(node) < min_maxima:
            min_maxima = n_graph.get(node)
    average_maxima = average_maxima/(len(maxima))
    stats = [len(maxima),min_maxima,average_maxima,n_graph.global_maxima()]
    stats = list(map(lambda x: str(x),stats))
    with open(args.output[0], 'a') as fout:
        fout.write(',(' + ';'.join(stats) + ')')
    print("Time:",time()-start)
    print("Local Minima Count:",len(n_graph.local_minima()))
    print("Local Maxima Count:",len(maxima))
    print("Minimum Local Maxima:", min_maxima)
    print("Average Local Maxima Value:", average_maxima)
    print("Maximum:",n_graph.global_maxima())
    print("Minimum:",n_graph.global_minima())
    print("Average:",n_graph.average())
    print("Number of Nodes:",len(n_graph))

def main_build_neighbor_monte_carlo(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]

    info = {}
    l_class = None
    if len(args.learn_conf) == 1:
        l_config = parse_configuration(args.learn_conf[0])
        info = l_config[1][0].info
        for type in agent_types:
            if l_config[1][0].type.lower() == type.string_id().lower():
                l_class = type
    else:
        info['clauses'] = 1
        info['literals'] = 1
        info['ranks'] = 5
        l_class = RankingPrefFormula

    results = []
    runs = 250
    start = time()
    agents = []
    for holder in config[1]:
        agents.append(make_agent(holder,agent_types,config[0]))
    ex_set = build_example_set_multi(agents, config[0])
    for i in range(runs):
        learner = l_class.random(config[0],info)
        # learner = SA.hillclimb(learner, ex_set, SA.evaluate_util)
        learner = SA.hillclimb(learner, ex_set, SA.evaluate_maximin)
        # results.append(SA.evaluate_util(learner, ex_set))
        results.append(SA.evaluate_maximin(learner, ex_set))
    average_maxima = 0.0
    for i in results:
        average_maxima += i
    average_maxima = average_maxima/(len(results))
    stats = [runs,min(results),average_maxima,max(results)]
    stats = list(map(lambda x: str(x),stats))
    with open(args.output[0], 'a') as fout:
        fout.write(',(' + ';'.join(stats) + ')')
    print("Time:",time()-start)

def main_hillclimb_rr(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]

    info = {}
    l_class = None
    if len(args.learn_conf) == 1:
        l_config = parse_configuration(args.learn_conf[0])
        info = l_config[1][0].info
        for type in agent_types:
            if l_config[1][0].type.lower() == type.string_id().lower():
                l_class = type
    else:
        info['clauses'] = 1
        info['literals'] = 1
        info['ranks'] = 5
        l_class = RankingPrefFormula

    runs = 400
    max_eval = 0
    start = time()
    stats = [100]
    agents = []
    for holder in config[1]:
        agents.append(make_agent(holder,agent_types,config[0]))
    ex_set = build_example_set_multi(agents, config[0])
    for i in range(runs):
        learner = l_class.random(config[0],info)
        learner = SA.hillclimb(learner, ex_set, SA.evaluate_util)
        # learner = SA.hillclimb(learner, ex_set, SA.evaluate_maximin)
        eval = SA.evaluate_util(learner, ex_set)
        # eval = SA.evaluate_maximin(learner, ex_set)
        if eval > max_eval:
            max_eval = eval
        if (i+1)%(stats[0]) == 0:
            stats.append(max_eval)
    stats = list(map(lambda x: str(x),stats))
    with open(args.output[0], 'a') as fout:
        fout.write(',(' + ';'.join(stats) + ')')
    print("Time:",time()-start)

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
#   agent is the original agent learned from.
#   learner is the learner to evaluate.
#   domain is a valid Domain object.
#
# Postcond:
#   Returns the proportion of examples in the ex_set.
def evaluate_rep_full(agent, learner, domain):
    count = 0
    total = 0
    for pair in domain.each_pair():
        rel = agent.build_example(pair[0],pair[1]).get_relation()
        if learner.compare(pair[0],pair[1]) == rel:
            count += 1
        total += 1
    return count/float(total)

# Precond:
#   agents is a list of the original agents learned from.
#   learner is the learner to evaluate.
#   domain is a valid Domain object.
#
# Postcond:
#   Returns the proportion of examples in the ex_set.
def evaluate_rep_full_multi(agents, learner, domain):
    count = 0
    pair_count = 0
    for agent in agents:
        for pair in domain.each_pair():
            # print(agent.build_example(pair[0],pair[1]))
            rel = agent.build_example(pair[0],pair[1]).get_relation()
            if learner.compare(pair[0],pair[1]) == rel:
                count += 1
            pair_count += 1
    return count/float(pair_count)

# Precond:
#   agents is a list of the original agents learned from.
#   learner is the learner to evaluate.
#   domain is a valid Domain object.
#
# Postcond:
#   Returns the proportion of examples in the ex_set.
def evaluate_rep_full_maximin(agents, learner, domain):
    current = None
    for agent in agents:
        count = 0
        pair_count = 0
        for pair in domain.each_pair():
            rel = agent.build_example(pair[0],pair[1]).get_relation()
            if learner.compare(pair[0],pair[1]) == rel:
                count += 1
            pair_count += 1
        count = count/float(pair_count)
        if current == None or count < current:
            current = count
    return current

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
#   Returns the proportion of examples in the ex_set for each agent.
def evaluate_multi(ex_set, learner):
    count = 0
    agent_counts = {}
    for agent in ex_set.get_agents():
        agent_counts[agent] = 0
    for example in ex_set.example_list():
        alts = example.get_alts()
        if learner.compare(alts[0],alts[1]) == example.get_relation():
            if example.get_agent() is not None:
                agent_counts[example.get_agent()] += 1
    agents = list(agent_counts.keys())
    agents.sort()
    result = []
    for agent in agents:
        total = ex_set.agent_count(agent)
        count = agent_counts[agent]
        if total > 0:
            result.append(count/float(total))
        else:
            result.append(0)
    return result

# Precond:
#   ex_set is the example set to evaluate.
#   learner is the learner to evaluate.
#   device is the device to run the tests on.
#
# Postcond:
#   Returns the proportion of examples in the ex_set for each agent.
def evaluate_cuda_multi(ex_set, learner, device=None):
    count = 0
    agent_counts = {}
    for agent in ex_set.get_agents():
        agent_counts[agent] = 0
    for i in range(len(ex_set)):
        inp,expect = ex_set[i]
        if device is not None:
            inp = inp.to(device)
        label = learner.forward_squash(inp)#.to(torch.device('cpu'))
        label = Relation.parse_label(label)
        if label.value == expect-2:
            if ex_set.get(i).get_agent() is not None:
                agent_counts[ex_set.get(i).get_agent()] += 1
    result = []
    agents = list(agent_counts.keys())
    agents.sort()
    for agent in agents:
        total = ex_set.agent_count(agent)
        count = agent_counts[agent]
        if total > 0:
            result.append(count/float(total))
        else:
            result.append(0)
    return result

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
#   pair is a pair of valid Alternative objects.
#
# Postcond:
#   Returns a tensor for input into a NN.
def prepare_pair(pair):
    inp = pair[0].values + pair[1].values
    inp = list(map(lambda x: float(x), inp))
    inp = torch.tensor(inp)
    del pair
    return inp

# Precond:
#   domain is a valid Domain object.
#   learner is a valid NeuralPreference object for the given domain.
#   device is a valid PyTorch device or None.
#
# Postcond:
#   Returns a preference graph generated by the preferences of the given
#   NN.
def build_NN_pref_graph(domain, learner, device=None):
    p_graph = PreferenceGraph(domain)
    equals = []
    for pair in domain.each_pair():
        inp = prepare_pair(pair)
        if device is not None:
            inp = inp.to(device)
        label = learner.forward_squash(inp)
        label = Relation.parse_label(label)
        if label == Relation.strict_preference():
            p_graph.arc(pair[0],pair[1])
        elif label == Relation.strict_dispreference():
            p_graph.arc(pair[1],pair[0])
        elif label == Relation.equal():
            found = False
            for item in equals:
                for entry in item:
                    if entry == pair[0] or entry == pair[1]:
                        found = True
                        item.append(pair[0])
                        item.append(pair[1])
                        break
                if found:
                    break
            if not found:
                equals.append([pair[0],pair[1]])
    for eq_set in equals:
        p_graph.share_arcs(eq_set)
    return p_graph

# Precond:
#   domain is a valid Domain object.
#   learner is a valid NeuralPreference object for the given domain.
#   agent is a valid Agent object for the given domain.
#   device is a valid PyTorch device or None.
#
# Postcond:
#   Returns the proportion of correctly classified example over the entire
#   pairwise comparison space of the domain.
def full_cuda_eval(domain, learner, agent, device=None):
    count = 0
    total = 0
    for pair in domain.each_pair():
        total += 1
        expect = agent.build_example(pair[0],pair[1]).relation
        inp = prepare_pair(pair)
        if device is not None:
            inp = inp.to(device)
        label = learner.forward_squash(inp)
        label = Relation.parse_label(label)
        if label == expect:
            count += 1
    return count/(float(total))

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

# Precond:
#   agent is a list of (Agent,size) pairs.
#   domain is the domain of the agent.
#
# Postcond:
#   Returns the example set for the agent.
def build_example_set_multi(agents, domain):
    result = ExampleSet()
    for agent in agents:
        pairs = domain.random_pair_set(agent[1])
        for pair in pairs:
            result.add_example(agent[0].build_example(pair[0],pair[1]))
        del pairs
    return result

def build_parser():
    parser = argparse.ArgumentParser(description="Automatically generate examples from randomly built synthetic agents.")
    parser.add_argument('-p', dest='problem', metavar='n', type=int, nargs=2, default=[1,1], help='Specified which problem/subproblem to run.')
    parser.add_argument('-l', dest='layers', metavar='n', type=int, nargs=1, default=[3], help='The number of neural net layers')
    parser.add_argument('-i', dest='learn_conf', metavar='filename', type=str, nargs=1, help='Name of the learner configuration file.', default='a.exs')
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser



if __name__=="__main__":
    args = build_parser().parse_args()
    if args.problem[0] == 1:
        if args.problem[1] == 1:
            main_learn_nn(args)
        elif args.problem[1] == 2:
            main_learn_joint_nn(args)
        elif args.problem[1] == 4:
            main_learn_nn_full(args) # <-- RETURN HERE
        else:
            print("Error: Unknown/Unavailable Subproblem.")
    elif args.problem[0] == 2:
        if args.problem[1] == 1:
            main_learn_lpm(args)
        elif args.problem[1] == 2:
            main_learn_joint_lpm(args)
        elif args.problem[1] == 3:
            main_learn_joint_lpm_mm(args)
        elif args.problem[1] == 4:
            main_learn_lpm_full(args)
        else:
            print("Error: Unknown/Unavailable Subproblem.")
    elif args.problem[0] == 3:
        if args.problem[1] == 1:
            main_learn_SA(args)
        elif args.problem[1] == 2:
            main_learn_joint_SA(args)
        elif args.problem[1] == 3:
            main_learn_joint_SA_mm(args)
        elif args.problem[1] == 4:
            main_learn_SA_full(args)
        else:
            print("Error: Unknown/Unavailable Subproblem.")
    elif args.problem[0] == 4:
        if args.problem[1] == 1:
            main_build_neighbor(args)
        elif args.problem[1] == 2:
            main_build_neighbor_monte_carlo(args)
        elif args.problem[1] == 3:
            main_hillclimb_rr(args)
        else:
            print("Error: Unknown/Unavailable Subproblem.")
    else:
        print("Error: Unknown/Unavailable Problem.")
