import argparse
import os
import torch
import gc
from math import sqrt
from time import time
from datetime import date
from examples.agent import Agent
from examples.example_set import ExampleSet
from examples.domain import Domain
from examples.portfolio_example import PortfolioExample
from examples.portfolio_example_set import PortfolioExampleSet
from examples.GCN_example import GCNExample
from examples.GCN_example_full import GCNExampleFull
from examples.GCN_example_set import GCNExampleSet
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
from neural.neural_preferences import train_neural_preferences, train_neural_preferences_curve, prepare_example
from neural.neural_portfolio import train_neural_portfolio
from neural.neural_portfolio_gcn import train_neural_portfolio_gcn
from annealing.simulated_annealing import learn_SA, learn_SA_mm
import annealing.simulated_annealing as SA
from uuid import uuid4


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

def single_run_curve(args, holder, agent_types, config, layers, learn_device):
    training = 0.0
    validation = 0.0
    agent = make_agent(holder,agent_types,config[0])
    ex_set = build_example_set(agent[0],agent[1],config[0])
    del agent
    for train, valid in ex_set.crossvalidation(5):
        # train.to_tensors(learn_device)
        # valid.to_tensors(learn_device)
        start = time()
        _,curve = train_neural_preferences_curve(train,layers,1000,config[0],learn_device)
        # learner.to(eval_device)
        print(time()-start)
        # pills.append('(' + str(training) + ';' + str(validation) + ')')
        curve_str = ';'.join(list(map(lambda x: str(x),curve)))
        with open(args.output[0],'a') as fout:
            fout.write(',(' + curve_str + ')')
        torch.cuda.empty_cache()
        del train
        del valid
    del ex_set

def single_run_full(args, holder, agent_types, config, layers, learn_device):
    training = 0.0
    validation = 0.0
    agent = make_agent(holder,agent_types,config[0])
    ex_set = build_example_set(agent[0],agent[1],config[0])
    for train, valid in ex_set.crossvalidation(5):
        # train.to_tensors(learn_device)
        # valid.to_tensors(learn_device)
        start = time()
        learner = train_neural_preferences(train,layers,3000,config[0],learn_device)
        # learner.to(eval_device)
        learner.eval()
        training = evaluate_cuda(train,learner,learn_device)
        validation = evaluate_cuda(valid,learner,learn_device)
        full_validation = full_cuda_eval(config[0],learner,agent[0],learn_device)
        # full_validation = -1
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

def main_learn_nn_curve(args):
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
        single_run_curve(args, holder, agent_types, config, layers, learn_device)

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

# main for neural network portfolio learning.
def main_nn_portfolio(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    learn_device = None
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    layers = [256 for i in range(max(0,args.layers[0]))]
    copies = 50
    types = []
    example_set = []
    # Generate example sets
    for _ in range(copies):
        for holder in config[1]:
            # build agent
            agent = make_agent(holder,agent_types,config[0])
            # build example_set
            ex_set = build_example_set(agent[0],agent[1],config[0])
            # convert to feature vector
            features = ex_set.get_feature_set()
            label = holder.type.lower()
            for i in range(len(agent_types)):
                if label == agent_types[i].string_id().lower():
                    if label not in types:
                        types.append(label)
            example_set.append(PortfolioExample(features,label))
            del agent
            del ex_set
            del features
    learning_set = PortfolioExampleSet(types)
    learning_set.add_example_list(example_set)
    layers.insert(0,6+config[0].pair_length())
    # layers.insert(0,6)
    layers.append(len(types))
    training = 0.0
    validation = 0.0
    for train, valid in learning_set.crossvalidation(5):
        start = time()
        print("START")
        learner = train_neural_portfolio(train,layers,2000,learn_device)
        print("END")
        learner.eval()
        training = evaluate_portfolio(train,learner,types,learn_device)
        validation = evaluate_portfolio(valid,learner,types,learn_device)
        print(time()-start)
        temp = ';'.join([str(training),str(validation)])
        with open(args.output[0],'a') as fout:
            fout.write(',(' + temp + ')')
        torch.cuda.empty_cache()
        del temp
        del learner
        del train
        del valid

# main for neural network portfolio learning.
def main_nn_full_portfolio(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    learn_device = None
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    layers = [256 for i in range(max(0,args.layers[0]))]
    copies = 50
    types = []
    example_set = []
    testing_set = []
    # Generate example sets
    for _ in range(copies):
        for holder in config[1]:
            # build agent
            agent = make_agent(holder,agent_types,config[0])
            # build example_set
            ex_set = build_full_example_set(agent[0],config[0])
            test_set = build_example_set(agent[0],agent[1],config[0])
            # convert to feature vector
            features = ex_set.get_feature_set()
            test_features = test_set.get_feature_set()
            label = holder.type.lower()
            for i in range(len(agent_types)):
                if label == agent_types[i].string_id().lower():
                    if label not in types:
                        types.append(label)
            example_set.append(PortfolioExample(features,label))
            testing_set.append(PortfolioExample(test_features,label))
            del agent
            del ex_set
            del test_set
            del features
    learning_set = PortfolioExampleSet(types)
    testing = PortfolioExampleSet(types)
    learning_set.add_example_list(example_set)
    testing.add_example_list(testing_set)
    layers.insert(0,6+config[0].pair_length())
    # layers.insert(0,6)
    layers.append(len(types))
    training = 0.0
    validation = 0.0
    total_training = 0.0
    total_validation = 0.0
    for train, valid in learning_set.crossvalidation(5):
        start = time()
        print("START")
        learner = train_neural_portfolio(train,layers,2000,learn_device)
        print("END")
        learner.eval()
        training = evaluate_portfolio(train,learner,types,learn_device)
        validation = evaluate_portfolio(valid,learner,types,learn_device)
        print(time()-start)
        total_training += training
        total_validation += validation
        torch.cuda.empty_cache()
        del learner
        del train
        del valid
    total_training = total_training/5.0
    total_validation = total_validation/5.0
    learner = train_neural_portfolio(learning_set, layers, 2000,learn_device)
    test_acc = evaluate_portfolio(testing, learner, types, learn_device)
    temp = ';'.join([str(total_training),str(total_validation),str(test_acc)])
    with open(args.output[0],'a') as fout:
        fout.write(',(' + temp + ')')

# main for GCN neural network portfolio learning.
def main_nn_portfolio_gcn(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    learn_device = None
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    lin_layers = [256 for i in range(max(1,args.layers[0]))]
    conv_layers = [128 for i in range(max(1,args.layers[0]))]
    copies = 50
    types = []
    example_set = []
    # Generate example sets
    for _ in range(copies):
        for holder in config[1]:
            # build agent
            agent = make_agent(holder,agent_types,config[0])
            # build example_set
            ex_set = build_example_set_gcn(agent[0],agent[1],config[0])
            label = holder.type.lower()
            for i in range(len(agent_types)):
                if label == agent_types[i].string_id().lower():
                    if label not in types:
                        types.append(label)
            # Convert example sets to needed GCN example sets and add them.
            example_set.append(GCNExample(ex_set, label))
            del agent
            del ex_set
    learning_set = GCNExampleSet(types)
    learning_set.add_example_list(example_set)
    if len(conv_layers) > 0 and len(lin_layers) > 0:
        conv_layers.insert(0,config[0].length())
        lin_layers.insert(0,conv_layers[-1])
        lin_layers.append(len(types))
    elif len(conv_layers) > 0 and len(lin_layers) == 0:
        conv_layers.insert(0, config[0].length())
        conv_layers.append(len(types))
    training = 0.0
    validation = 0.0
    total_training = 0.0
    total_validation = 0.0
    for train, valid in learning_set.crossvalidation(5):
        start = time()
        learner = train_neural_portfolio_gcn(train,conv_layers,lin_layers,1000,learn_device)
        print(time()-start)
        learner.eval()
        training = evaluate_portfolio_gcn(train,learner,types,learn_device)
        validation = evaluate_portfolio_gcn(valid,learner,types,learn_device)
        total_training += training
        total_validation += validation
        torch.cuda.empty_cache()
        del learner
        del train
        del valid
    total_training = total_training/5.0
    total_validation = total_validation/5.0
    temp = ';'.join([str(total_training),str(total_validation)])
    with open(args.output[0],'a') as fout:
        fout.write(',(' + temp + ')')

def main_nn_portfolio_gcn_full(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    learn_device = None
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    lin_layers = [128 for i in range(max(1,args.layers[0]))]
    conv_layers = [64 for i in range(max(1,args.layers[0]))]
    copies = 50
    types = []
    example_set = []
    testing_set = []
    # Generate example sets
    ex_start = time()
    for holder in config[1]:
        for _ in range(copies):
            # build agent
            agent = make_agent(holder,agent_types,config[0])
            # build example_set
            test_set = build_example_set_gcn(agent[0],agent[1],config[0])
            label = holder.type.lower()
            # Faster generation of example by not going through intermediaries.
            example_set.append(GCNExampleFull(config[0],agent[0],label))
            for i in range(len(agent_types)):
                if label == agent_types[i].string_id().lower():
                    if label not in types:
                        types.append(label)
            # Convert example sets to needed GCN example sets and add them.
            testing_set.append(GCNExample(test_set, label))
            del agent
            del test_set
    print("Examples generated:",time()-ex_start)
    learning_set = GCNExampleSet(types)
    testing = GCNExampleSet(types)
    learning_set.add_example_list(example_set)
    testing.add_example_list(testing_set)
    if len(conv_layers) > 0 and len(lin_layers) > 0:
        conv_layers.insert(0,config[0].length())
        lin_layers.insert(0,conv_layers[-1])
        lin_layers.append(len(types))
    elif len(conv_layers) > 0 and len(lin_layers) == 0:
        conv_layers.insert(0, config[0].length())
        conv_layers.append(len(types))
    training = 0.0
    validation = 0.0
    total_training = 0.0
    total_validation = 0.0
    for train, valid in learning_set.crossvalidation(5):
        start = time()
        learner = train_neural_portfolio_gcn(train,conv_layers,lin_layers,1000,learn_device)
        print(time()-start)
        learner.eval()
        training = evaluate_portfolio_gcn(train,learner,types,learn_device)
        validation = evaluate_portfolio_gcn(valid,learner,types,learn_device)
        total_training += training
        total_validation += validation
        torch.cuda.empty_cache()
        del learner
        del train
        del valid
    total_training = total_training/5.0
    total_validation = total_validation/5.0
    learner = train_neural_portfolio_gcn(learning_set, conv_layers, lin_layers,1000,learn_device)
    test_acc = evaluate_portfolio_gcn(testing, learner, types, learn_device)
    temp = ';'.join([str(total_training),str(total_validation),str(test_acc)])
    with open(args.output[0],'a') as fout:
        fout.write(',(' + temp + ')')

def learn_RPF(ex_set, domain):
    info = {}
    info['clauses'] = 2
    info['literals'] = 2
    info['ranks'] = 7
    resets = 5
    best_model = None
    best_perf = 0
    for i in range(resets):
        learner = RankingPrefFormula.random(domain, info)
        learner = learn_SA(learner, ex_set)
        perf = evaluate_rep(ex_set,learner)
        if perf > best_perf or best_model is None:
            best_model = learner
            best_perf = perf
    return best_model

def learn_ASO(ex_set, domain):
    info = {}
    info['ranks'] = 3
    info['rules'] = 3
    info['formulas'] = 7
    info['clauses'] = 2
    info['literals'] = 2
    resets = 5
    best_model = None
    best_perf = 0
    for i in range(resets):
        learner = ASO.random(domain, info)
        learner = learn_SA(learner, ex_set)
        perf = evaluate_rep(ex_set,learner)
        if perf > best_perf or best_model is None:
            best_model = learner
            best_perf = perf
    return best_model

def main_build_portfolio_training_set(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    copies = 500
    types = []
    example_set = []
    ex_set_size = 0
    # Generate example sets
    for _ in range(copies):
        for holder in config[1]:
            # build agent
            agent = make_agent(holder,agent_types,config[0])
            # build example_set
            ex_set = build_example_set(agent[0],agent[1],config[0])
            ex_set_size = max(ex_set_size,agent[1])
            # convert to feature vector
            features = ex_set.get_feature_set()
            label = holder.type.lower()
            for i in range(len(agent_types)):
                if label == agent_types[i].string_id().lower():
                    if label not in types:
                        types.append(label)
            example_set.append(PortfolioExample(features,label))
            del agent
            del ex_set
            del features
    learning_set = PortfolioExampleSet(types)
    learning_set.add_example_list(example_set)
    with open(args.output[0],'w') as fout:
        fout.write("#Portfolio Prediction Training Set\n")
        fout.write("#Built on " + str(date.today())+"\n")
        fout.write("#Total types: "+str(len(types))+"\n")
        fout.write("#Total Number of Example Sets: "+str(len(example_set))+"\n")
        fout.write("#Largest Number of Examples per Example Set: "+str(ex_set_size)+"\n")
        learning_set.to_file(fout)

def main_build_portfolio_training_set_smart(args):
    config = parse_configuration(args.config[0])
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    copies = 25
    types = []
    example_set = []
    ex_set_size = 0
    learning_algorithms = {}
    learning_algorithms['LPM'] = LPM.learn_greedy
    learning_algorithms['RPF'] = learn_RPF
    learning_algorithms['ASO'] = learn_ASO
    # Generate example sets
    for _ in range(copies):
        for holder in config[1]:
            # build agent
            agent = make_agent(holder,agent_types,config[0])
            # build example_set
            ex_set = build_example_set(agent[0],agent[1],config[0])
            ex_set_size = max(ex_set_size,agent[1])
            # convert to feature vector
            features = ex_set.get_feature_set()
            best_perf = 0
            best_model = 'LPM'
            for lang in learning_algorithms.keys():
                model = learning_algorithms[lang](ex_set,domain)
                perf = evaluate_rep(ex_set,model)
                if perf > best_perf:
                    best_model = lang
                data_line.append(perf)
            label = best_model.lower()
            for i in range(len(agent_types)):
                if label == agent_types[i].string_id().lower():
                    if label not in types:
                        types.append(label)
            example_set.append(PortfolioExample(features,label))
            del agent
            del ex_set
            del features
    learning_set = PortfolioExampleSet(types)
    learning_set.add_example_list(example_set)
    with open(args.output[0],'w') as fout:
        fout.write("#Portfolio Prediction Training Set\n")
        fout.write("#Built on " + str(date.today())+"\n")
        fout.write("#Total types: "+str(len(types))+"\n")
        fout.write("#Total Number of Example Sets: "+str(len(example_set))+"\n")
        fout.write("#Largest Number of Examples per Example Set: "+str(ex_set_size)+"\n")
        learning_set.to_file(fout)

def main_build_portfolio_classifier(args):
    ex_set = parse_protfolio_example_file(args.ex_file[0])
    labels = ex_set.labels
    learn_device = None
    if torch.cuda.is_available():
        learn_device = torch.device('cuda')
    else:
        learn_device = torch.device('cpu')
    layers = [256 for i in range(max(0,args.layers[0]))]
    layers.insert(0,6)
    layers.append(len(labels))
    best_classifier = None
    best_perf = 0.0
    for i in range(10):
        start = time()
        classifier = train_neural_portfolio(ex_set,layers,3000,learn_device)
        print(time()-start)
        training_perf = evaluate_portfolio(ex_set,classifier,labels,learn_device)
        torch.cuda.empty_cache()
        if training_perf > best_perf or best_classifier is None:
            best_classifier = classifier
            best_perf = training_perf
    print(best_perf)
    best_classifier.to(torch.device('cpu'))
    torch.save(best_classifier,args.nn_file[0])
    with open(args.labels_file[0],'w') as fout:
        fout.write(','.join(labels))
    return

def main_test_portfolio_learner(args):
    # Build learner selector dictionary
    learning_algorithms = {}
    learning_algorithms['LPM'] = LPM.learn_greedy
    learning_algorithms['RPF'] = learn_RPF
    learning_algorithms['ASO'] = learn_ASO
    test_order = list(learning_algorithms.keys())
    test_order.sort()
    # Load classifier and labels
    classifier = torch.load(args.nn_file[0])
    classifier.to(torch.device('cpu'))
    classifier.eval()
    labels = []
    with open(args.labels_file[0],'r') as fin:
        data = fin.read()
        data = data.split("\n")
        for line in data:
            line = line.strip()
            if line[0] != "#":
                labels = line.split(',')
                labels = list(map(lambda x: x.strip(),labels))
                break

    # Generate test example sets
    agent_types = [LPM, RankingPrefFormula, PenaltyLogic, WeightedAverage, CPnet, CLPM, LPTree, ASO]
    examples_per_set = 100
    testing_example_sets = []
    domain = Domain(8,[2 for i in range(8)])
    repeat = 25
    print('Building examples...')
    for agent_type in agent_types:
        for i in range(repeat):
            agent = make_agent_scratch(agent_type, domain)
            ex_set = build_example_set(agent,examples_per_set,domain)
            ex_set.lang = agent_type.string_id()
            testing_example_sets.append(ex_set)
    # Classify and learn
    data = []
    print('Testing...')
    des_str = ["Original,Choice,Portfolio"]
    des_str.extend(test_order)
    des_str = ','.join(des_str)
    with open(args.output[0], 'w') as fout:
        fout.write(des_str+"\n")
    for ex_set in testing_example_sets:
        start = time()
        data_line = [ex_set.lang]
        features = torch.tensor(ex_set.get_feature_set())
        label = classifier.forward_squash(features)
        current = 0
        for j in range(len(label)):
            if label[j] > label[current]:
                current = j
        prediction = labels[current]
        data_line.append(prediction)
        # Select learning algorithm
        learn_alg = None
        if prediction in learning_algorithms:
            learn_alg = learning_algorithms[prediction]
        else:
            learn_alg = learn_ASO
        model = learn_alg(ex_set,domain)
        perf = evaluate_rep(ex_set,model)
        data_line.append(perf)
        print('Done prediction: ', time()-start)
        # Test all learning algorithms separately
        for lang in test_order:
            model = learning_algorithms[lang](ex_set,domain)
            perf = evaluate_rep(ex_set,model)
            data_line.append(perf)
        data_line = list(map(lambda x: str(x),data_line))
        data_line = ','.join(data_line)
        with open(args.output[0],'a') as fout:
            fout.write(data_line+"\n")
        print('Done all: ',time()-start)
        # data.append(data_line)
    return


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
#   ex_set is the example set to evaluate.
#   learner is the learner to evaluate.
#   device is the device to run the tests on.
#
# Postcond:
#   Returns the proportion of correctly decided examples in the ex_set.
def evaluate_portfolio(ex_set, learner, labels, device=None):
    fname = 'mistakes/mistakes_' + str(uuid4()) + '.txt'
    fout = open(fname, 'w')
    fout.close()
    del fout
    count = 0
    for i in range(len(ex_set)):
        inp,expect = ex_set[i]
        if device is not None:
            inp = inp.to(device)
        label = learner.forward_squash(inp)#.to(torch.device('cpu'))
        current = 0
        for j in range(len(label)):
            if label[j] > label[current]:
                current = j
        des_str = "(" + str(labels[current]) + ',' + str(labels[expect]) + ")\n"
        if current == expect:
            count += 1
        else:
            with open(fname, 'a') as fout:
                fout.write(des_str + "\n")
        del inp
        del expect
        del label
    return count/float(len(ex_set))

# Precond:
#   ex_set is the example set to evaluate.
#   learner is the learner to evaluate.
#   device is the device to run the tests on.
#
# Postcond:
#   Returns the proportion of correctly decided examples in the ex_set.
def evaluate_portfolio_gcn(ex_set, learner, labels, device=None):
    fname = 'mistakes/mistakes_' + str(uuid4()) + '.txt'
    fout = open(fname, 'w')
    fout.close()
    del fout
    count = 0
    for i in range(len(ex_set)):
        inp,expect = ex_set[i]
        if device is not None:
            inp = inp.to(device)
        label = learner.forward_squash(inp)[0]#.to(torch.device('cpu'))
        current = 0
        for j in range(len(label)):
            if label[j] > label[current]:
                current = j
        des_str = "(" + str(labels[current]) + ',' + str(labels[expect]) + ")\n"
        if current == expect:
            count += 1
        else:
            with open(fname, 'a') as fout:
                fout.write(des_str + "\n")
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
#   type is a valid Agent type class specifier.
#   domain is the domain of the agents.
#
# Postcond:
#   Returns tuple of:
#       1) A valid random Agent object of the type specfied by agent.
#       2) The number of examples to create.
def make_agent_scratch(type, domain):
    return Agent(type.random(domain,type.random_info(domain)),domain)

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
#   agent is a valid Agent object.
#   size is the number of examples in the example set.
#   domain is the domain of the agent.
#
# Postcond:
#   Returns the example set for the agent.
def build_example_set_gcn(agent, size, domain):
    result = ExampleSet()
    alts = domain.sample(int(sqrt(2*size)))
    for i in range(len(alts)):
        for j in range(i+1,len(alts)):
            result.add_example(agent.build_example(alts[i],alts[j]))
    del alts
    return result

# Precond:
#   agent is a valid Agent object.
#   domain is the domain of the agent.
#
# Postcond:
#   Returns an example set which includes all possible pairs for the agent.
def build_full_example_set(agent, domain):
    result = ExampleSet()
    for pair in domain.each_pair():
        result.add_example(agent.build_example(pair[0],pair[1]))
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

# Precond:
#   filename is the name of a valud portfolio example file.
#
# Postcond:
#   Parses a portfolio example file and returns a PortfolioExampleSet object.
def parse_protfolio_example_file(filename):
    ex_set = []
    labels = []
    with open(filename,'r') as fin:
        line = fin.readline()
        while line:
            line = line.strip()
            if line[0] != '#':
                if line.startswith("LABELS"):
                    line = line.split(':')[1].strip()
                    labels = line.split(',')
                elif line.startswith("EX"):
                    line = line.split(':')[1].strip()
                    line = line.split(',')
                    label = line[0].strip()
                    vec = line[1].strip()
                    vec = vec.split(';')
                    vec = list(map(lambda x: float(x),vec))
                    ex_set.append(PortfolioExample(vec,label))
            line = fin.readline()
    result = PortfolioExampleSet(labels)
    result.add_example_list(ex_set)
    return result

def build_parser():
    parser = argparse.ArgumentParser(description="Automatically generate examples from randomly built synthetic agents.")
    parser.add_argument('-p', dest='problem', metavar='n', type=int, nargs=2, default=[1,1], help='Specified which problem/subproblem to run.')
    parser.add_argument('-e', dest='ex_file', metavar='filename', type=str, nargs=1, help='Name of the example file to use for training.', default='a.ex')
    parser.add_argument('-l', dest='layers', metavar='n', type=int, nargs=1, default=[3], help='The number of neural net layers')
    parser.add_argument('-i', dest='learn_conf', metavar='filename', type=str, nargs=1, help='Name of the learner configuration file.', default='a.config')
    parser.add_argument('-o', dest='output', metavar='filename', type=str, nargs=1, help='Name of the output file.', default='a.exs')
    parser.add_argument('-n', dest='nn_file', metavar='filename', type=str, nargs=1, help='NN file (I/O as needed).', default=['a.nn'])
    parser.add_argument('-m', dest='labels_file', metavar='filename', type=str, nargs=1, help='Label file (I/O as needed).', default=['a.labels'])
    parser.add_argument('config', metavar='filename', type=str, nargs=1, help="The config file to use.")
    return parser



if __name__=="__main__":
    args = build_parser().parse_args()
    # Neural network problems
    if args.problem[0] == 1:
        if args.problem[1] == 1:
            main_learn_nn(args)
        elif args.problem[1] == 2:
            main_learn_joint_nn(args)
        elif args.problem[1] == 4:
            main_learn_nn_full(args) # <-- RETURN HERE
        elif args.problem[1] == 5:
            # Neural network learning curve analysis
            main_learn_nn_curve(args)
        else:
            print("Error: Unknown/Unavailable Subproblem.")
    # LPM problems
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
    # Simulated Annealing problems
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
    # Misc. Problems.
    elif args.problem[0] == 4:
        if args.problem[1] == 1:
            main_build_neighbor(args)
        elif args.problem[1] == 2:
            main_build_neighbor_monte_carlo(args)
        elif args.problem[1] == 3:
            main_hillclimb_rr(args)
        elif args.problem[1] == 4:
            main_nn_portfolio(args)
        elif args.problem[1] == 5:
            main_nn_full_portfolio(args)
        elif args.problem[1] == 6:
            main_nn_portfolio_gcn(args)
        elif args.problem[1] == 7:
            main_nn_portfolio_gcn_full(args)
        else:
            print("Error: Unknown/Unavailable Subproblem.")
    elif args.problem[0] == 5:
        if args.problem[1] == 1:
            main_build_portfolio_training_set(args)
        elif args.problem[1] == 2:
            main_build_portfolio_classifier(args)
        elif args.problem[1] == 3:
            main_test_portfolio_learner(args)
        elif args.problem[1] == 4:
            main_build_portfolio_training_set_smart(args)
        else:
            print("Error: Unknown/Unavailable Subproblem.")

    else:
        print("Error: Unknown/Unavailable Problem.")
