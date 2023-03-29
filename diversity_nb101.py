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


parser = argparse.ArgumentParser(description='Diversity')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./nasbench_full.tfrecord',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/', type=str, help='folder to save results')
parser.add_argument('--nasspace', default='nasbench101', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--evaluate_size', default=256, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)


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


if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'


# Create a pickle file with curated information about the data set
search_space_config = {}
for i, (uid, _) in enumerate(searchspace):
    print (i, uid)
    
    model_data = searchspace.query_model(uid)
    
    search_space_config[i] = {}
    search_space_config[i]['config'] = model_data['module_operations']
    search_space_config[i]['train_params'] = model_data['trainable_parameters'] 
    search_space_config[i]['train_params'] = model_data['training_time']
    search_space_config[i]['c10-train'] = model_data['train_accuracy']*100
    search_space_config[i]['c10-val'] = model_data['validation_accuracy']*100
    search_space_config[i]['c10-test'] = searchspace.get_accuracy(uid,acc_type)#due to multiple runs, get the best #model_data['test_accuracy']*100
    #quit()

# Save file
with open('search_space_config_nb101.pickle', 'wb') as handle:
    pickle.dump(search_space_config, handle, protocol=pickle.HIGHEST_PROTOCOL)


def count_total_occurrences(list_ops, search_space_config):
    def count_occurrences(list_ops, cell):
        '''
        Count the occurences of each operation in a given cell
        '''
        ops_dict = dict.fromkeys(list_ops, []) # create dict with empty lists to store accs
        for op in list_ops: #[conv3x3,max,conv1x1]
            sum = 0
            for in_value_op in cell:
                if in_value_op == op:
                    sum+=1
            ops_dict[op] = sum
        return ops_dict
    def count_ocurrences_per_dataset(list_ops, search_space_config, dataset):
        ocurrences = dict((op,{}) for op in list_ops)#dict.fromkeys(list_ops, None) # start with all ops, sum=0
        for idx, (_, cell_values) in enumerate(search_space_config.items()):
            cell_occurences = count_occurrences(list_ops, cell_values['config'])
            for layer, count in cell_occurences.items():
                if count not in ocurrences[layer]:
                    ocurrences[layer][count] = []
                ocurrences[layer][count] += [cell_values[dataset]]
        return ocurrences

    dsets_total_occurences =  dict((dset,{}) for dset in datasets)#dict.fromkeys(datasets, {})
    for dataset in datasets:
        ocurrences = count_ocurrences_per_dataset(list_ops, search_space_config, dataset)
        dsets_total_occurences[dataset] = ocurrences
    return dsets_total_occurences


file_to_read = open("search_space_config_nb101.pickle", "rb")
search_space_config = pickle.load(file_to_read)
nb_101_list_ops = ['input', 'output', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
datasets = ['c10-train', 'c10-val', 'c10-test']
total_occurences=count_total_occurrences(nb_101_list_ops, search_space_config)
with open('layer_occurences_nb101.pickle', 'wb') as handle:
    pickle.dump(total_occurences, handle, protocol=pickle.HIGHEST_PROTOCOL)
