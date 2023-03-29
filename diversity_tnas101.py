import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from tqdm import trange
from statistics import mean
import time
import collections
from utils import add_dropout
from operator import itemgetter
import pickle
import json

parser = argparse.ArgumentParser(description='Diversity')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./transnas-bench_v10141024.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/', type=str, help='folder to save results')
parser.add_argument('--nasspace', default='transnas101', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--evaluate_size', default=256, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--search_type', default='micro', type=str, help="type of the search space for transnas-bench-101: micro/macro")


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)

ops = {'0': 'None', '1': 'Skip Connection', '2': 'Conv. 1x1', '3': 'Conv. 3x3'}
'''
search_space_config = {}
all_models = searchspace.api.all_arch_dict[searchspace.ss]
for idx, model in enumerate(all_models, 0): 
    #example: 64-41414-3_33_333, where "3_33_333"
    # - node0: input tensor
    # - node1: Conv3x3( node0 ) # 3
    # - node2: Conv3x3( node0 ) + Conv3x3( node1 ) # 33
    # - node3: Conv3x3( node0 ) + Conv3x3( node1 ) + Conv3x3( node2 ) # 333
    # where 0 - None; 1 - Skip_connect; 2 - Conv1x1; 3 - Conv3x3

    model_config = model.split("-")[2].split("_")
    config = []
    for node in model_config:
        for op in list(node):
            config.append(ops[op])

    search_space_config[idx] = {}
    search_space_config[idx]['config'] = config

    all_results_unstructured = searchspace.train_and_eval(idx).all_results
    for task in searchspace.get_datasets():
        search_space_config[idx][task] = {}
        search_space_config[idx][task]['model_info'] = all_results_unstructured[task]['model_info']
        for metric in searchspace.get_metrics(task):
            search_space_config[idx][task][metric] = searchspace.api.get_single_metric(model, task, metric, mode='best')
            #print(task, metric)
            #print(searchspace.api.get_single_metric(model, task, metric, mode='best'))
            #print(searchspace.api.get_best_epoch_status(model, task, metric=metric))
with open('search_space_config_transnas101.pickle', 'wb') as handle:
    pickle.dump(search_space_config, handle, protocol=pickle.HIGHEST_PROTOCOL)


def count_total_occurrences(list_ops, search_space_config, tasks):
    def count_occurrences(list_ops, cell):
        #Count the occurences of each operation in a given cell
        
        ops_dict = dict.fromkeys(list_ops, []) # create dict with empty lists to store accs
        for op in list_ops:
            sum = 0
            for in_value_op in cell: # [list of ops]
                if in_value_op == op: 
                    sum+=1
            ops_dict[op] = sum
        return ops_dict

    def count_ocurrences_per_metric(list_ops, search_space_config, task):
        ocurrences = dict((op,{}) for op in list_ops)#dict.fromkeys(list_ops, None) # start with all ops, sum=0
        for idx, (_, cell_values) in enumerate(search_space_config.items()):
            cell_occurences = count_occurrences(list_ops, cell_values['config'])
            for layer, count in cell_occurences.items():
                for metric in searchspace.get_metrics(task):
                    if count not in ocurrences[layer]:
                        ocurrences[layer][count] = {}
                    #print(cell_values[dataset])
                    if metric not in ocurrences[layer][count]:
                        ocurrences[layer][count][metric] = []
                    #print (cell_values[task][metric])
                    ocurrences[layer][count][metric] += [cell_values[task][metric]]
        return ocurrences

    dsets_total_occurences =  dict((task,{}) for task in tasks)#dict.fromkeys(datasets, {})
    for task in tasks:
        ocurrences = count_ocurrences_per_metric(list_ops, search_space_config, task)
        dsets_total_occurences[task] = ocurrences
    return dsets_total_occurences

list_ops = ['None', 'Skip Connection', 'Conv. 1x1', 'Conv. 3x3']
tasks = searchspace.get_datasets()
total_occurences=count_total_occurrences(list_ops, search_space_config, tasks)
with open('layer_occurences_transnas101.pickle', 'wb') as handle:
    pickle.dump(total_occurences, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
all_models = searchspace.api.all_arch_dict[searchspace.ss]
for idx, model in enumerate(all_models, 0):
    print(model)
    print(searchspace.get_network(model))
    quit()