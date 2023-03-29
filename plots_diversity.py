
from turtle import color
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.figure_factory as ff
from scipy.stats import norm
from scipy import stats
from itertools import product
import re
import json
import os



nb_201_list_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
#datasets = ['c10-val', 'c10-test', 'c100-val', 'c100-test', 'imgnet16120-val', 'imgnet16120-test']
datasets = ['c10-val', 'c100-val', 'imgnet16120-val']

file_to_read = open("layer_occurences_nb201.pickle", "rb")
layer_ocurrences = pickle.load(file_to_read)
#print(layer_ocurrences['imgnet16120-test']['none'])
#print(layer_ocurrences['c10-test']['none'][5])
#print(len(layer_ocurrences['c10-test']['none'][1]))

file_to_read = open("search_space_config.pickle", "rb")
search_space_config = pickle.load(file_to_read)


plt.rcParams.update({'font.size': 20,
                    'legend.fontsize': 16})


if not os.path.exists('./results'):
    os.makedirs('./results')

'''
def generate_boxplots_per_op():
    for dataset in datasets:
        for op in nb_201_list_ops:
            plt.clf()
            df=pd.DataFrame.from_dict(layer_ocurrences[dataset][op], orient='index').transpose()
            df = df.reindex(sorted(df.columns), axis=1) #order columns ascending
            print(df)
            sns.boxplot(data=df, palette="Set1", width=0.5)
            if "imgnet" in dataset:
                plt.ylim(0, 50)
            elif "c100" in dataset:
                plt.ylim(0, 80)
            else:
                plt.ylim(0, 100)
            plt.xlabel('# Occurrences')
            plt.ylabel('Test Accuracy (%)')
            plt.grid('on', linestyle='--')
            plt.savefig("./results/"+dataset+"_"+op+".pdf", bbox_inches='tight')
'''

def generate_boxplots_per_op():
    list_ops = {'none': 'None', 'skip_connect': 'Skip Connection', 'nor_conv_1x1': 'Conv. 1x1', 'nor_conv_3x3' : 'Conv. 3x3', 'avg_pool_3x3':'Avg. Pool 3x3'}
    for dataset in datasets:
        rows_list = []
        for op, ocurrences in layer_ocurrences[dataset].items():
            for n_ocurrences, values in ocurrences.items():
                for value in values:
                    rows_list.append({'operation': list_ops[op], 'value':value, 'Ocurrences':n_ocurrences})
        df = pd.DataFrame(rows_list)
        plt.clf()
        plt.figure(figsize=(20,5))
        sns.boxplot(x='operation', y='value', hue='Ocurrences', data=df, palette="Set1", width=0.75)
        if "imgnet" in dataset:
            plt.ylim(0, 50)
        elif "c100" in dataset:
            plt.ylim(0, 80)
        else:
            plt.ylim(0, 95)
            plt.yticks([0,20,40,60,80,95])
        plt.xlabel('Operation Type')
        plt.ylabel('Accuracy (%)')
        plt.grid('on', linestyle='--')
        legend = plt.legend(edgecolor="black", loc = (0.75,0.1))
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((0, 0, 0, 0.0))
        print("yo")
        plt.savefig("./results/"+dataset+"_boxplot"+".pdf", bbox_inches='tight')


def generate_spider_graph(datasets, list_ops, layer_ocurrences):
    values = []
    for dataset in datasets: #c10, c10-val ....
        #df = dict((op,{}) for op in list_ops)
        op_values = []
        for op in list_ops: #conv3x3, conv1x1, ..
            list_values = []
            for ocurrence in layer_ocurrences[dataset][op]:
                if ocurrence == 0:
                    continue
                list_values += layer_ocurrences[dataset][op][ocurrence]
            op_values.append(round(np.mean(list_values),3))
        values.append(op_values)
    # print(list_ops) #['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    list_ops = ["None", "Skip Connection", "Conv. 1x1", "Conv. 3x3", "Avg. Pool 3x3"]
    fig = go.Figure(
        data=[
            #C-10
            go.Scatterpolar(r=[*values[0],values[0][0]], theta=[*list_ops,list_ops[0]], fill='toself', marker = dict(color = '#1f77b4')),#, name='CIFAR-10
            #C-100
            #go.Scatterpolar(r=[*values[1],values[1][0]], theta=[*list_ops,list_ops[0]],fill='toself', marker = dict(color = '#9467bd')), #, name='CIFAR-100'
            #IMGNET16-120
            #go.Scatterpolar(r=[*values[2],values[2][0]], theta=[*list_ops,list_ops[0]],fill='toself', marker = dict(color = '#2ca02c')) #, name='ImageNet16-120'
        ],
        layout=go.Layout(
            #title=go.layout.Title(text='Restaurant comparison'),
            #C-10 val
            polar={'radialaxis': {'visible': True, 'range': list([82,86]), 'tickvals': [82,83,84,85,86], 'tickfont' : {'size':18}}, 'angularaxis' : {'tickfont' : {'size':20}}}, 
            #C-100 val
            #polar={'radialaxis': {'visible': True, 'range': list([60,64]), 'tickvals': [60,61,62,63,64], 'tickfont' : {'size':18}}, 'angularaxis' : {'tickfont' : {'size':20}}}, 
            #IMGNET16120-val
            #polar={'radialaxis': {'visible': True, 'range': list([32,37]), 'tickvals': [32,33,34,35,36,37], 'tickfont' : {'size':18}}, 'angularaxis' : {'tickfont' : {'size':20}}}, 
            #ALL
            #polar={'radialaxis': {'visible': True, 'range': list([32,86]), 'tickvals': [32,33,34,35,36,37,60,61,62,63,64,82,83,84,85,86]}}, 
            showlegend=False,

        )
    )

    #pyo.plot(fig)
    fig.write_image(f"./results/c10_spider_plot.pdf")



def generate_bell_graph(datasets, list_ops, layer_ocurrences):
    accs_per_dataset = dict((dataset,list()) for dataset in datasets)
    for cell, values in layer_ocurrences.items():
        for dataset in datasets:
            #print(values[dataset])
            accs_per_dataset[dataset] += [values[dataset]]

    # Group data together
    hist_data = []
    for dataset in datasets:
        hist_data.append(accs_per_dataset[dataset])

    group_names = ['CIFAR-10', 'CIFAR-100', 'ImageNet16-120']
    # Create distplot with custom bin_size
    colors = ['#1f77b4', '#9467bd', '#2ca02c']

    # Dist &  bell curve with plotly
    #fig = ff.create_distplot(hist_data, group_names, show_hist=True, show_rug=False, bin_size=.2, colors=colors)
    #fig.write_image("bell_curve.pdf")

    # Dist &  bell curve with seaborn & matplot for consistency
    sns.distplot(hist_data[0], hist=True, rug=False, color=colors[0], label=group_names[0])
    sns.distplot(hist_data[1], hist=True, rug=False, color=colors[1], label=group_names[1])
    sns.distplot(hist_data[2], hist=True, rug=False, color=colors[2], label=group_names[2])
    plt.yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    legend = plt.legend(edgecolor="black")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0.0))
    #plt.legend()
    plt.xlabel("Accuracy (%)")
    plt.show()
    plt.savefig("./results/dist.pdf", bbox_inches='tight')



def get_all_permutations_dict(list_ops, perm_n=2):
        all_ops_permutations = {}
        for permutation_ops in product(list_ops, repeat=perm_n):#in permutations(list_ops, 2):
                all_ops_permutations["|".join(list(permutation_ops))] = []
            #all_ops_permutations[first_op+"|"+second_op] = []
        return all_ops_permutations


def generate_op_sequence_matrix_all_datasets(datasets, list_ops, search_space_config, perm_n=2):
    '''
    generate operation combination matrix
    perm_n defines the number of operation permutations (e.g., perm_n=2 will get none|none...,
    but perm_n=4 will get none|none|none|none)
    '''
    def generate_op_sequence_matrix(dataset, list_ops, search_space_config, perm_n=2):
        all_ops_permutations = get_all_permutations_dict(list_ops, perm_n)
        for cell_idx, values in search_space_config.items():
            cell_ops_sequence=' '.join(re.sub("~[0-9]", "+",values['config']['arch_str'].replace("|","")).split("+")).split()    
            #print(cell_ops_sequence)
            last_op = cell_ops_sequence[:perm_n-1]
            for op in cell_ops_sequence[perm_n-1:]:
                combined_ops = "|".join(last_op)+"|"+op
                #combined_ops = last_op+"|"+op
                all_ops_permutations[combined_ops] += [values[dataset]]
                last_op = combined_ops.split("|")[-perm_n+1:]
        return all_ops_permutations

    combinations_per_dataset = {}
    for dataset in datasets:
        combinations_per_dataset[dataset] = generate_op_sequence_matrix(dataset, list_ops, search_space_config, perm_n=2)
        # calculate mean and std
        print(dataset)
        for op, value in combinations_per_dataset[dataset].items():
            mean_std = (f'{np.mean(value):.3f} +/- {np.std(value):.3f}')
            combinations_per_dataset[dataset][op] = mean_std
            #print(f'{op} -> {np.mean(value):.3f} +/- {np.std(value):.3f}')
            #print(combinations_per_dataset[dataset][op])

        # Save to file
        json.dump(combinations_per_dataset[dataset], open("./results/"+dataset+"_"+str(perm_n)+"permutations.json", 'w' ))
        print("Files generated and stored.")


def get_top_combinations(datasets, list_ops, search_space_config, max_perms=6, top_k = (10,)):
    for dataset in datasets: # iterate over all datasets
        dataset_perms = {}
        for perm_n in range (1, max_perms+1): #iterate over all perms combinations (none.none, none.none.none , ...)
            all_ops_permutations = get_all_permutations_dict(list_ops, perm_n)
            for cell_idx, values in search_space_config.items():
                cell_ops_sequence=' '.join(re.sub("~[0-9]", "+",values['config']['arch_str'].replace("|","")).split("+")).split()
                #if cell_ops_sequence[0] == "avg_pool_3x3" and cell_ops_sequence[1] == "avg_pool_3x3" and cell_ops_sequence[2] == "avg_pool_3x3" and cell_ops_sequence[3] == "avg_pool_3x3" and cell_ops_sequence[4] == "avg_pool_3x3" and cell_ops_sequence[5] == "avg_pool_3x3":
                #    print ("está feito")
                #    quit()
                last_op = cell_ops_sequence[:perm_n-1]
                for op in cell_ops_sequence[perm_n-1:]:
                    if perm_n > 1:
                        combined_ops = "|".join(last_op)+"|"+op
                    else:
                        combined_ops = op
                    #combined_ops = last_op+"|"+op
                    all_ops_permutations[combined_ops] += [values[dataset]]
                    last_op = combined_ops.split("|")[-perm_n+1:]
            dataset_perms.update(all_ops_permutations)
        #for idx, perm_list in enumerate(dataset_perms):
        for key,value in dataset_perms.items():
            dataset_perms[key] = {'mean': np.mean(value), 'std': np.std(value)}
        dict_stored_as_list = sorted(dataset_perms.items(), key=lambda x: x[1]['mean'], reverse=True)

        print(dataset)
        for k in top_k:
            # Save to file
            json.dump(dict_stored_as_list[:k], open(f'./results/{dataset}_top{k}_combinations.json', 'w' ))
            print("Files generated and stored.")
            print (f'top {k}')
            for idx, op_comb in enumerate(dict_stored_as_list):
                if (idx > k):
                    break
                print(op_comb)
            print ("\n")
        #print(dataset_perms)

def mean_acc_per_op_per_position(datasets, list_ops, search_space_config):
    for dataset in datasets: # iterate over all datasets
        ops = dict((op,{1:list(),2:list(),3:list(),4:list(),5:list(),6:list()}) for op in list_ops)
        for idx_1, (cell_idx, values) in enumerate(search_space_config.items()):
            cell_ops_sequence=' '.join(re.sub("~[0-9]", "+",values['config']['arch_str'].replace("|","")).split("+")).split()    
            for idx, op in enumerate(cell_ops_sequence,1):
                ops[op][idx] += [values[dataset]]
        
        for key,value in ops.items():
            for key_position, value_position in value.items():
                ops[key][key_position] = {'mean': np.mean(value_position), 'std': np.std(value_position)}
        
        json.dump(ops, open(f'./results/{dataset}_operations_positions_within_cell .json', 'w' ))
        print("Files generated and stored.")

def rank_through_datasets_nb201(datasets, search_space_config, k_rank=50):
    dict_stored_as_list = sorted(search_space_config.items(), key=lambda x: x[1]['c10-val'], reverse=True)
    for dataset in datasets[::-1]: #finish in c-10
            dict_stored_as_list = sorted(dict_stored_as_list, key=lambda x: x[1][dataset], reverse=True)
            for idx, item in enumerate(dict_stored_as_list,1):
                dict_stored_as_list[idx-1][1]['config'][dataset+"_rank"]=idx

    print(dict_stored_as_list[0])
    values_to_plot = []
    for idx, item in enumerate(dict_stored_as_list):
        if idx >= k_rank:
            break
        list_aux = []
        for dataset in datasets[::]:
            list_aux.append(item[1]['config'][dataset+'_rank'])
        values_to_plot.append(list_aux)
    
    #df = pd.DataFrame(values_to_plot, columns=['CIFAR-10', 'CIFAR-100', 'ImageNet16-120'])
    #print(df)
    #sns.lineplot(data=df)
    #colors = pl.cm.jet(np.linspace(0,1,k_rank))
    colors =  plt.cm.twilight_shifted( np.linspace(0,1,k_rank) )
    for idx, value in enumerate(values_to_plot):
        plt.plot(['CIFAR-10', 'CIFAR-100', 'ImageNet16-120'], value, color=colors[idx])
        #print(idx, value)
    #plt.legend()
    plt.xlabel("Dataset")
    plt.ylabel("Rank")
    plt.ylim(0,k_rank)
    plt.show()
    plt.savefig("./results/rank.pdf", bbox_inches='tight')

def correlation_between_datasets_nb201(datasets, search_space_config):
    df = pd.DataFrame.from_dict(search_space_config).T[datasets]
    df = df.apply(pd.to_numeric)
    df.columns = ['CIFAR-10', 'CIFAR-100', 'ImageNet16-120']
    #df.drop('config')
    print(df.head)
    corr = df.corr(method='kendall')
    plt.clf()
    #corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    sns.heatmap(corr, 
            annot=True, vmax=1, vmin=0.8, center=0.9, cmap='YlOrRd')
    plt.xticks(rotation=10)
    plt.savefig("./results/corr_nb201.pdf", bbox_inches='tight')

#generate_boxplots_per_op()
#generate_spider_graph(['c10-val', 'c100-val', 'imgnet16120-val'], nb_201_list_ops, layer_ocurrences)
generate_bell_graph(['c10-val', 'c100-val', 'imgnet16120-val'], nb_201_list_ops, search_space_config)
#generate_op_sequence_matrix_all_datasets(['c10-val', 'c100-val', 'imgnet16120-val'], nb_201_list_ops, search_space_config)
#get_top_combinations(['c10-val', 'c100-val', 'imgnet16120-val'], nb_201_list_ops, search_space_config)
#mean_acc_per_op_per_position(['c10-val', 'c100-val', 'imgnet16120-val'], nb_201_list_ops, search_space_config)
#rank_through_datasets_nb201(['c10-val', 'c100-val', 'imgnet16120-val'], search_space_config)
#correlation_between_datasets_nb201(['c10-val', 'c100-val', 'imgnet16120-val'], search_space_config)