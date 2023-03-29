from turtle import color
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pl
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.figure_factory as ff
from scipy.stats import norm
from itertools import product
import re
import json
import os
from brokenaxes import brokenaxes

import nasspace
import argparse

plt.rcParams.update({'font.size': 20,
                    'legend.fontsize': 16})

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
searchspace = nasspace.get_search_space(args) # to get metrics per dataset

def generate_boxplots_per_op(metrics_to_evaluate):
    for task in tasks:
        metrics = set(searchspace.get_metrics(task)) & set(metrics_to_evaluate)
        for metric in metrics:
            rows_list = []
            for op, ocurrences in layer_ocurrences[task].items():
                for n_ocurrences, values in ocurrences.items():
                    for value in values[metric]:
                        rows_list.append({'operation': op, 'value':value, 'Occurrences':n_ocurrences})
            df = pd.DataFrame(rows_list)
            #print(df.head)
            plt.clf()
            plt.figure(figsize=(20,5))
            sns.boxplot(x='operation', y='value', hue='Occurrences', data=df, palette="Set1", width=0.75)
            plt.xlabel('Operation Type')
            plt.ylabel(metrics_correspondence[metric])
            legend = plt.legend(edgecolor="black", loc = "lower right")
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((0, 0, 0, 0.0))
            plt.grid('on', linestyle='--')
            plt.savefig("./results/transnas101/"+task+"_"+metric+"_boxplot"+".pdf", bbox_inches='tight')


def generate_spider_graph(tasks, list_ops, layer_ocurrences, metrics_to_evaluate):
    for idx_task, task in enumerate(tasks):
        #values = []
        metrics = set(searchspace.get_metrics(task)) & set(metrics_to_evaluate)
        for idx_metric, metric in enumerate(metrics):
            #df = dict((op,{}) for op in list_ops)
            op_values = []
            for op in list_ops: #conv3x3, conv1x1, ..
                list_values = []
                for ocurrence, values in layer_ocurrences[task][op].items():
                    if ocurrence == 0: #ignore if there are no occurences of the op
                        continue
                    #print(layer_ocurrences[task][op][ocurrence][0][metric])
                    #for n_ocurrences, values in layer_ocurrences[task][op][ocurrence].items():
                    for value in values[metric]:
                        list_values += [value]

                op_values.append(round(np.mean(list_values),3))
            #values.append(op_values)
            # print(list_ops) #['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
            ticks = np.linspace(np.min(op_values), np.max(op_values)+0.001,5,endpoint=False)
            ticks = [np.min(op_values)-(ticks[1]-ticks[0])] + list(ticks)
            ticks = [ '%.2f' % elem for elem in ticks ]
            range_values = [ticks[0], np.max(op_values)]

            colors = ['#1f77b4', '#9467bd', '#2ca02c', '#FF5733', '#DAF7A6', '#717D7E', '#3498DB','#F08080','#DFFF00', '#000000']
            fig = go.Figure(
                data=[
                    go.Scatterpolar(r=[*op_values,op_values[0]], theta=[*list_ops,list_ops[0]], fill='toself', marker = dict(color = colors[idx_task])),#, name='CIFAR-10
                ],
                layout=go.Layout(
                    polar={'radialaxis': {'visible': True, 'range': range_values, 'tickvals': ticks, 'tickfont' : {'size':18}}, 'angularaxis' : {'tickfont' : {'size':20}}}, 
                    showlegend=False,
                )
            )
            #pyo.plot(fig)
            fig.write_image(f"./results/transnas101/"+task+"_"+metric+"_spider_plot.pdf")


def generate_bell_graph_all_tasks_combined(tasks, search_space_config, evaluate_metrics):
    task_metrics = {
        'class_object': 'valid_top1',
        'class_scene': 'valid_top1',
        'jigsaw': 'valid_top1',
        'room_layout': 'valid_neg_loss',
        'autoencoder': 'valid_ssim',
        'normal': 'valid_ssim',
        'segmentsemantic' : 'valid_mIoU',
    }

    task_correspondence = {
        'class_object': 'Object Class.',
        'class_scene': 'Scene Class.',
        'jigsaw': 'Jigsaw',
        'room_layout': 'Room Layout',
        'autoencoder': 'AutoEncoder',
        'normal': 'Surf. Normal',
        'segmentsemantic': 'Sem. Segmentation',
    }
    
    hist_data = {}
    for task in task_metrics.keys():
        metric = task_metrics[task]
        accs_per_metric = []
        for cell, values in search_space_config.items():
                #print(values)
                value = values[task][metric]
                #print(value)
                #print(metric)
                if 'ssim' in metric.lower() or 'mIoU' in metric.lower() or 'neg_loss' in metric.lower():
                    if value < 0:
                        value = value*-100
                    else:
                        value = value*100
                #print(value)
                accs_per_metric += [value]

        # Group data together
        if not task in hist_data:
            hist_data[task] = []
        hist_data[task] = accs_per_metric

    colors = ['#1f77b4', '#9467bd', '#2ca02c', '#FF5733', '#DAF7A6', '#717D7E', '#3498DB','#F08080','#DFFF00', '#000000']
        # Dist &  bell curve with seaborn & matplot for consistency
    plt.figure(figsize=(10,8))
    print(hist_data.keys())
    for i, task in enumerate(hist_data.keys()):
        #print (task)
        sns.distplot(hist_data[task], hist=True, rug=False, color=colors[i], label=task_correspondence[task])
    plt.yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    #plt.legend(fancybox=True, framealpha=0, edgecolor="black")
    legend = plt.legend(edgecolor="black")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0.0))
    plt.xlabel("Performance")
    plt.show()
    plt.savefig("./results/transnas101/dist.pdf", bbox_inches='tight')


def generate_bell_graph(tasks, search_space_config, evaluate_metrics):
    for task in tasks:
        #print(searchspace.get_metrics(task))
        #print(evaluate_metrics)
        metrics = set(searchspace.get_metrics(task)) & set(evaluate_metrics)
        accs_per_metric = dict((metric,list()) for metric in metrics)
        #print(metrics)
        for cell, values in search_space_config.items():
            for metric in metrics:
                #print(values)
                accs_per_metric[metric] += [values[task][metric]]

        # Group data together
        hist_data = []
        for metric in metrics:
            hist_data.append(accs_per_metric[metric])

        #group_names = ['CIFAR-10', 'CIFAR-10-Test', 'CIFAR-10-Train']
        # Create distplot with custom bin_size
        #colors = ['#1f77b4', '#9467bd', '#2ca02c']
        colors = ['#1f77b4', '#9467bd', '#2ca02c', '#FF5733', '#DAF7A6', '#717D7E', '#3498DB','#F08080','#DFFF00', '#000000']

        # Dist &  bell curve with plotly
        #fig = ff.create_distplot(hist_data, group_names, show_hist=True, show_rug=False, bin_size=.2, colors=colors)
        #fig.write_image("bell_curve.pdf")

        # Dist &  bell curve with seaborn & matplot for consistency
        plt.clf()
        legend_metric = []
        if task  == 'segmentsemantic':
            f, (ax,ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

            for idx, metric in enumerate(metrics):
                if idx == 1:
                    sns.distplot(hist_data[idx], hist=True, rug=False, color=colors[idx], label=metrics_correspondence[metric], ax=ax)
                else:
                    sns.distplot(hist_data[idx], hist=True, rug=False, color=colors[idx], label=metrics_correspondence[metric], ax=ax2)
                legend_metric += [metrics_correspondence[metric]]
                        # zoom-in / limit the view to different portions of the data
            ax.set_xlim(0, 35)  # outliers only
            ax2.set_xlim(90, 95)  # most of the data
            ax.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.yaxis.set_visible(False)
            ax.yaxis.tick_left()
            d = .01 # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((1-d,1+d), (-d,+d), **kwargs)
            ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d,+d), (1-d,1+d), **kwargs)
            ax2.plot((-d,+d), (-d,+d), **kwargs)
            f.text(0.5, 0.04, '/'.join(legend_metric), ha='center')

            handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            # plot the legend
            #plt.legend(handles=handles2.append(handles))  
            #print(plt.gca().get_legend_handles_labels())     
            f.legend(handles, labels, bbox_to_anchor=(0.3, 0.37, 0.5, 0.5), ncol=1, borderaxespad=1)
            
            #f.xlabel('/'.join(legend_metric))
        else:
            for idx, metric in enumerate(metrics):
                sns.distplot(hist_data[idx], hist=True, rug=False, color=colors[idx], label=metrics_correspondence[metric])
                legend_metric += [metrics_correspondence[metric]]
            plt.xlabel('/'.join(legend_metric))
        
            plt.legend()
        #print(task)

        plt.show()
        plt.savefig("./results/transnas101/"+task+"_"+"dist.pdf", bbox_inches='tight')

def get_top_cells(tasks, evaluate_metrics, top_k = (10,), search_type='micro'):
    for task in tasks:
        for metric in set(searchspace.get_metrics(task)) & set(evaluate_metrics):
            for k in top_k:
                json.dump(
                    searchspace.api.get_best_archs(task, metric, search_type, k),
                    open(f'./results/transnas101/{task}_{metric}_top{k}cells.json', 'w' ))

def get_top_cells_ranking(tasks, metrics_to_evaluate, search_space_config, top_k = 10):
    task_metrics = {'class_object': 'valid_top1', 'class_scene': 'valid_top1',
                    'jigsaw': 'valid_top1', 'room_layout': 'valid_neg_loss',
                    'autoencoder': 'valid_ssim', 'normal': 'valid_ssim',
                    'segmentsemantic' : 'valid_mIoU',
                }

    dict_stored_as_list = sorted(search_space_config.items(), key=lambda x: x[1]['class_object']['valid_top1'], reverse=True)
    for task, metric in task_metrics.items():
        #metrics = set(searchspace.get_metrics(task)) & set(metrics_to_evaluate)
        #for metric in metrics:
        dict_stored_as_list = sorted(dict_stored_as_list, key=lambda x: x[1][task][metric], reverse=True)
        for idx, item in enumerate(dict_stored_as_list,1):
            dict_stored_as_list[idx-1][1][task][metric] = {'value':dict_stored_as_list[idx-1][1][task][metric],
                                                            'rank':idx}#[dataset+"_rank"]=idx
    #print((dict_stored_as_list[0][1][task][metric]))

    for task, metric in task_metrics.items():
        dict_stored_as_list = sorted(dict_stored_as_list, key=lambda x: x[1][task][metric]['value'], reverse=True)
        #Save to file
        json.dump(dict_stored_as_list[:top_k],
            open("./results/transnas101/"+task+"_"+metric+"_top_cells_curated.json", 'w' ))



def generate_op_sequence_matrix_all_datasets(tasks, list_ops, search_space_config, metrics_to_evaluate, perm_n=2):
    '''
    generate operation combination matrix
    perm_n defines the number of operation permutations (e.g., perm_n=2 will get none|none...,
    but perm_n=4 will get none|none|none|none)
    '''
    def get_all_permutations_dict(list_ops, perm_n=2):
        all_ops_permutations = {}
        for permutation_ops in product(list_ops, repeat=perm_n):#in permutations(list_ops, 2):
                all_ops_permutations["|".join(list(permutation_ops))] = []
            #all_ops_permutations[first_op+"|"+second_op] = []
        return all_ops_permutations

    def generate_op_sequence_matrix(task, list_ops, search_space_config, metric, perm_n=2):
        all_ops_permutations = get_all_permutations_dict(list_ops, perm_n)
        for cell_idx, values in search_space_config.items():
            cell_ops_sequence=values['config']#' '.join(re.sub("~[0-9]", "+",values['config']['arch_str'].replace("|","")).split("+")).split()    
            #print(cell_ops_sequence)
            last_op = cell_ops_sequence[:perm_n-1]
            for op in cell_ops_sequence[perm_n-1:]:
                combined_ops = "|".join(last_op)+"|"+op
                #combined_ops = last_op+"|"+op
                all_ops_permutations[combined_ops] += [values[task][metric]]
                last_op = combined_ops.split("|")[-perm_n+1:]
        return all_ops_permutations

    for task in tasks:
        #combinations_per_task = {}
        metrics = set(searchspace.get_metrics(task)) & set(metrics_to_evaluate)
        for metric in metrics:
            combinations_per_task = generate_op_sequence_matrix(task, list_ops, search_space_config, metric, perm_n=2)
            # calculate mean and std
            #print(task)
            for op, value in combinations_per_task.items():
                mean_std = (f'{np.mean(value):.3f} +/- {np.std(value):.3f}')
                combinations_per_task[op] = mean_std
                #print(f'{op} -> {np.mean(value):.3f} +/- {np.std(value):.3f}')
                #print(combinations_per_dataset[dataset][op])

            # Save to file
            json.dump(combinations_per_task,
                open("./results/transnas101/"+task+"_"+metric+"_"+str(perm_n)+"permutations.json", 'w' ))
            #print("Files generated and stored.")


def get_inter_task_ranking(tasks, metrics_to_evaluate, search_space_config, top_k = 10, k_rank=50):
    dict_stored_as_list = sorted(search_space_config.items(), key=lambda x: x[1][tasks[0]][metrics_to_evaluate[0]], reverse=True)
    for task in tasks:
        metrics = set(searchspace.get_metrics(task)) & set(metrics_to_evaluate)
        for metric in metrics:
            dict_stored_as_list = sorted(dict_stored_as_list, key=lambda x: x[1][task][metric], reverse=True)
            for idx, item in enumerate(dict_stored_as_list,1):
                dict_stored_as_list[idx-1][1][task][metric] = {'value':dict_stored_as_list[idx-1][1][task][metric],
                                                                'rank':idx}#[dataset+"_rank"]=idx
    #print((dict_stored_as_list[0][1][task][metric]))
    for task in tasks:
        metrics = set(searchspace.get_metrics(task)) & set(metrics_to_evaluate)
        for metric in metrics:
            dict_stored_as_list = sorted(dict_stored_as_list, key=lambda x: x[1][task][metric]['value'], reverse=True)
            # Save to file
            json.dump(dict_stored_as_list[:top_k],
                open("./results/transnas101/"+task+"_"+metric+"_top_cells.json", 'w' ))

    values_to_plot = []
    task_metrics = {'class_object': 'valid_top1', 'class_scene': 'valid_top1',
                    'jigsaw': 'valid_top1', 'room_layout': 'valid_neg_loss',
                    'autoencoder': 'valid_ssim', 'normal': 'valid_ssim',
                    'segmentsemantic' : 'valid_mIoU',
                }
    dict_stored_as_list = sorted(dict_stored_as_list, key=lambda x: x[1][tasks[0]][task_metrics[tasks[0]]]['value'], reverse=True)
    for idx, item in enumerate(dict_stored_as_list):
        if idx >= k_rank:
            break
        list_aux = []
        #print (item)
        for task, metric in task_metrics.items():
            list_aux.append(item[1][task][metric]['rank'])
        values_to_plot.append(list_aux)

    colors =  plt.cm.twilight_shifted( np.linspace(0,1,k_rank) )
    plt.clf()
    plt.figure(figsize=(10,8))
    for idx, value in enumerate(values_to_plot):
        #print(value)
        plt.plot(['Object Class.', 'Scene Class.', 'Jigsaw', 'Room Layout',
         'AutoEncoder', 'Surf. Normal', 'Sem. Segmentation'], value, color=colors[idx])
        #print(idx, value)
    #plt.legend()
    #plt.xlabel("Task")
    plt.ylabel("Rank")
    plt.xticks(rotation=20)
    plt.ylim(0,k_rank)
    plt.show()
    plt.savefig("./results/transnas101/rank.pdf", bbox_inches='tight')


def mean_acc_per_op_per_position(tasks, list_ops, search_space_config):
    task_metrics = {'class_object': 'valid_top1', 'class_scene': 'valid_top1',
                'jigsaw': 'valid_top1', 'room_layout': 'valid_neg_loss',
                'autoencoder': 'valid_ssim', 'normal': 'valid_ssim',
                'segmentsemantic' : 'valid_mIoU',
            }
    for task, metric in task_metrics.items():
        ops = dict((op,{1:list(),2:list(),3:list(),4:list(),5:list(),6:list()}) for op in list_ops)
        for idx_1, (cell_idx, values) in enumerate(search_space_config.items()):
            #print(values)
            for idx, op in enumerate(values['config'],1):
                ops[op][idx] += [values[task][metric]]
        
        for key,value in ops.items():
            for key_position, value_position in value.items():
                ops[key][key_position] = {'mean': np.mean(value_position), 'std': np.std(value_position)}
        
        json.dump(ops, open(f'./results/transnas101/{task}_operations_positions_within_cell.json', 'w' ))
        print("Files generated and stored.")


def correlation_between_datasets_tnb201(tasks, metrics, search_space_config):
    task_metrics = {'class_object': 'valid_top1', 'class_scene': 'valid_top1',
                'jigsaw': 'valid_top1', 'room_layout': 'valid_neg_loss',
                'autoencoder': 'valid_ssim', 'normal': 'valid_ssim',
                'segmentsemantic' : 'valid_mIoU',
            }
    new_search_space_config = {}
    for item, values in search_space_config.items():
        new_search_space_config[item] = {}
        for task, metric in task_metrics.items():
            new_search_space_config[item][task] = search_space_config[item][task][metric]
    
    df = pd.DataFrame.from_dict(new_search_space_config).T[tasks]
    df = df.apply(pd.to_numeric)
    df.columns = ['Cls. Object', 'Cls. Scene', 'Jigsaw', 'Room Layout', 'Autoencoding', 'Surf. Normal', 'Sem. Segment.']
    #df['Room Layout'] = df['Room Layout'].apply(lambda x:x* -100)
    print(df.head)
    corr = df.corr(method='kendall')
    plt.clf()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, 
            annot=True, vmax=1, vmin=0, center=0.5, cmap='YlOrRd')
    plt.xticks(rotation=20)
    plt.savefig("./results/transnas101/corr_tnb201.pdf", bbox_inches='tight')



try:
    os.makedirs("./results/transnas101/", exist_ok=False)
except:
    pass #directory already created
list_ops = ['None', 'Skip Connection', 'Conv. 1x1', 'Conv. 3x3']
tasks = ['class_object', 'class_scene', 'jigsaw', 'room_layout', 'autoencoder', 'normal', 'segmentsemantic'] #tasks
file_to_read = open("layer_occurences_transnas101.pickle", "rb")
layer_ocurrences = pickle.load(file_to_read)
#print(layer_ocurrences['class_scene']['None'][0])

file_to_read = open("search_space_config_transnas101.pickle", "rb")
search_space_config = pickle.load(file_to_read)
#print(search_space_config[0])

metrics_correspondence = {'top1': 'Acc. (%)', 'top5': 'Top 5 Acc. (%)', 'valid_l1_loss': 'L1 Loss', 'valid_ssim': 'SSIM',
                         'valid_neg_loss':'L2 Loss', 'valid_acc':'Acc. (%)', 'valid_mIoU': 'mIoU', 'valid_top1':'Acc. (%)',
                         'valid_top5': 'Top 5 Acc. (%)', 'valid_loss' : 'Validation Loss', 'time_elapsed': 'Time Elapsed',
                         'train_top1': 'Train Acc. (%)', 'train_top5': 'Train Top 5 Acc. (%)', 'train_neg_loss':'Train L2 Loss', 'train_acc': 'Train Acc. (%)',
                         'train_mIoU':'Train mIoU', 'train_l1_loss': 'Train L1 Loss', 'train_ssim': 'Train SSIM', 'train_loss': 'Train Loss',
                         'test_top1': 'Test Acc. (%)', 'test_top5': 'Test Top 5 Acc. (%)', 'test_neg_loss':'Test L2 Loss', 'test_acc': 'Test Acc. (%)',
                         'test_mIoU':'Test mIoU', 'test_l1_loss': 'Test L1 Loss', 'test_ssim': 'Test SSIM', 'test_loss': 'Test Loss'}
metrics_to_evaluate    = ['valid_top1', 'valid_l1_loss', 'valid_ssim', 'valid_neg_loss', 'valid_acc', 'valid_mIoU'] #'valid_loss'

generate_boxplots_per_op(metrics_to_evaluate)
#generate_spider_graph(tasks, list_ops, layer_ocurrences, metrics_to_evaluate)
#generate_bell_graph(tasks, search_space_config, metrics_to_evaluate)
#generate_bell_graph_all_tasks_combined(tasks, search_space_config, metrics_to_evaluate)
#generate_op_sequence_matrix_all_datasets(tasks, list_ops, search_space_config, metrics_to_evaluate, perm_n=2)
#get_top_cells(tasks, metrics_to_evaluate, top_k = (10,), search_type='micro')
get_top_cells_ranking(tasks, metrics_to_evaluate, search_space_config, top_k = 10)
#get_inter_task_ranking(tasks, metrics_to_evaluate, search_space_config)
#mean_acc_per_op_per_position(None, list_ops, search_space_config)
#correlation_between_datasets_tnb201(tasks, metrics_to_evaluate, search_space_config)
#print(search_space_config[0])