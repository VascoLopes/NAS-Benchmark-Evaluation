from models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import NASBench201API as API
from nasbench import api as nasbench101api
from nas_101_api.model import Network
from nas_101_api.model_spec import ModelSpec
from transnas_api.api import TransNASBenchAPI
import itertools
import random
import numpy as np
from models.cell_searchs.genotypes import Structure
from copy import deepcopy
from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.anynet import AnyNet
from pycls.models.nas.genotypes import GENOTYPES, Genotype
import json
import torch
from transnas_api.lib.models.net_infer.net_macro import MacroNet #transnas101
import transnas_api.tools.utils as trans_utils
import argparse
from argparse import Namespace

from transnas_api.lib.models.task_models.feedforward import FeedForwardNet
from transnas_api.lib.models.task_models.siamese import SiameseNet
from transnas_api.lib.models.task_models.decoder import FFDecoder, SiameseDecoder, GenerativeDecoder, SegmentationDecoder
from transnas_api.lib.models.task_models.encoder import FFEncoder
from transnas_api.lib.models.task_models.discriminator import Discriminator
from transnas_api.lib.models.task_models.gan import GAN
from transnas_api.lib.models.task_models.segmentation import Segmentation


class Transnas101:
    #ops = {'0': 'None', '1': 'Skip Connection', '2': 'Conv. 1x1', '3': 'Conv. 3x3'}
    def __init__(self, dataset, apiloc, args):
        self.dataset = dataset
        self.api = TransNASBenchAPI(apiloc)

        self.important_metrics_per_task = {
            'class_object': ['valid_top1', 'test_top1', 'valid_top5', 'test_top5'],
            'class_scene': ['valid_top1', 'test_top1', 'valid_top5', 'test_top5'],
            'room_layout': ['valid_loss', 'test_loss', 'valid_neg_loss', 'test_neg_loss'],
            'jigsaw': ['valid_top1', 'test_top1', 'valid_top5', 'test_top5'],
            'autoencoder': ['valid_l1_loss', 'test_l1_loss', 'valid_ssim', 'test_ssim'],
            'segmentsemantic': ['valid_acc', 'test_acc', 'valid_mIoU', 'test_mIoU'],
            'normal': ['valid_l1_loss', 'test_l1_loss', 'valid_ssim', 'test_ssim'],
        }

        try:
            self.ss = args.search_type
        except:
            self.ss = 'micro'
        try:
            self.metric = args.metric
        except:
            self.metric = 'valid_top1'
        try:
            self.eval_epoch = args.eval_epoch
        except:
            self.eval_epoch = 24
        self.search_space_archs = self.api.all_arch_dict[self.ss]

        self.config_dir={
            'class_object': "./transnas_api/configs/train_from_scratch/class_object/",
            'class_scene': "./transnas_api/configs/train_from_scratch/class_scene/",
            'room_layout': "./transnas_api/configs/train_from_scratch/room_layout/",
            'jigsaw': "./transnas_api/configs/train_from_scratch/jigsaw/",
            'autoencoder': "./transnas_api/configs/train_from_scratch/autoencoder/",
            'segmentsemantic': "./transnas_api/configs/train_from_scratch/segmentsemantic/",
            'normal': "./transnas_api/configs/train_from_scratch/normal/",
        }

        self.target_dim = {
            'class_object': 75,
            'class_scene': 47,
            'room_layout': 9,
            'jigsaw': 1000,
            'autoencoder': (256, 256),
            'segmentsemantic': (256, 256),
            'normal': (256, 256),
        }

        self.max_epochs = { #epochs go from [0->max[ e.g., 25: [0,25[
            'class_object': 24, 
            'class_scene': 24,
            'room_layout': 24,
            'jigsaw': 9,
            'autoencoder': 29,
            'segmentsemantic': 29,
            'normal': 29,
        }
        
    def __iter__(self):
        for uid in self.search_space_archs:
            network = self.get_network(uid)
            yield uid, network
    def __getitem__(self, index):
        try:
            return self.api.__getitem__(index)
        except:
            return index #already in correct format
    def __len__(self):
        #ss = api.search_spaces # list of search space names
        return len(self.api.all_arch_dict[self.ss]) # {search_space : list_of_architecture_names}
    def get_arch_str(self, uid):
        try:
            return uid.arch_str
        except:
            return uid
    def get_datasets(self): # get all the tasks
        return self.api.task_list
    def get_metrics(self, task):
        return self.api.metrics_dict[task]
    def random_arch(self):
        return self.get_arch_str(random.choice(self.search_space_archs))
        #return random.randint(0, len(self)-1)
    def get_config(self, uid):
        return self.api.get_arch_result(uid)
        #return arch_result[task]['model_info'][info]
    def get_spec(self, uid):
        return self.api.index2arch(uid)
    def get_final_accuracy(self, uid, acc_type, trainval):
        if trainval == False:
            try:
                metric = self.metric.replace("valid", "test")
            except:
                metric = self.metric
        else:
            metric = self.metric
        return self.api.get_best_epoch_status(uid, self.dataset, metric=metric)[metric]
    def get_12epoch_accuracy(self, uid, epoch, trainval, traincifar10=False): #epoch -> acc_type
        return self.api.get_epoch_status(uid, self.dataset, epoch)[self.metric]
    def get_training_time(self, uid, epoch=12):
        return self.api.get_epoch_status(uid, self.dataset, epoch)['time_elapsed']
    def query_epoch(self, uid, epoch, args=None):
        return self.api.get_epoch_status(uid, self.dataset, epoch=epoch)
    def train_and_eval(self, arch, dataname, acc_type, trainval=True, traincifar10=False):
        arch = self.get_arch_str(arch)
        twelveth_epochs = 11
        if twelveth_epochs > self.max_epochs[self.dataset]:
            twelveth_epochs = self.max_epochs[self.dataset]
        time = self.get_training_time(arch, twelveth_epochs) #starts at 0
        acc12 = self.get_12epoch_accuracy(arch, twelveth_epochs, trainval, traincifar10) #TODO
        acc = self.get_final_accuracy(arch, acc_type, trainval)
        #time = self.get_training_time(arch, self.eval_epoch)
        #return acc12, acc, time
        return acc, acc, time
    '''
    def get_network(self, uid):
        uid = self.get_arch_str(uid)
        #config = "64-41414-1_11_111"
        network = MacroNet(uid, structure='full')#.cuda()
        return network
    '''
    def get_cfg(self, uid='resnet50'):
        # Get Arguments TODO: change this....
        #encoder_str defines the arch config
        # $config_dir/$task_name/ -> config for the dataset
        '''
        parser = argparse.ArgumentParser(description='.')
        parser.add_argument('--cfg_dir', type=str, help='directory containing config.py file')
        parser.add_argument('--encoder_str', type=str, default='resnet50')
        parser.add_argument('--nopause', dest='nopause', action='store_true')
        parser.add_argument('--seed', type=int, default=666)
        parser.add_argument('--ddp', action='store_true')
        parser.set_defaults(nopause=True)
        args = parser.parse_args()
        '''
        args = Namespace(cfg_dir= self.config_dir[self.dataset],
                encoder_str= uid,
                nopause= False,
                seed= 666,
                ddp= False,
        )
        #args.cfg_dir = self.config_dir[self.dataset]
        device_list = list(range(torch.cuda.device_count()))

        trans_utils.prepare_seed_cudnn(args.seed)
        cfg = trans_utils.setup_config(args, len(device_list))
        return cfg, device_list
    def get_network(self, uid='resnet50'):
        rank = 0
        #cfg, device_list = self.get_cfg(uid)
        # setup config
        #trans_utils.log_cfg(cfg, logger, nopause=False)
        #model = trans_utils.setup_model(cfg, device_list, rank, ddp=False)

        encoder = FFEncoder(uid, task_name=self.dataset).network
        decoder_input_dim = (2048, 16, 16) if uid == 'resnet50' else encoder.output_dim
        if self.dataset == 'class_object' or self.dataset== 'class_scene' or self.dataset == 'room_layout':
            decoder = FFDecoder(decoder_input_dim, self.target_dim[self.dataset])
            model = FeedForwardNet(encoder, decoder)
        elif self.dataset == 'jigsaw':
            decoder = SiameseDecoder(decoder_input_dim, self.target_dim[self.dataset], num_pieces=9)
            model = SiameseNet(encoder, decoder)
        elif self.dataset == 'normal' or self.dataset == 'autoencoder':
            decoder = GenerativeDecoder(decoder_input_dim, self.target_dim[self.dataset])
            model = GAN(encoder, decoder, Discriminator())
        elif self.dataset == 'segmentsemantic':
            decoder = SegmentationDecoder(decoder_input_dim, self.target_dim[self.dataset],
                                         target_num_channel=17)
            model = Segmentation(encoder, decoder)
        return model
    def mutate_arch(self, arch): #TODO
        #print(arch)
        arch = self.get_arch_str(arch)
        if self.ss == 'micro':
            child_arch = deepcopy(arch.split("-")[2])
            child_arch = list(itertools.chain(*child_arch.split("_")))
            #print(child_arch)
            #child_arch = [x for xs in xss for x in child_arch]
            node_id = random.randint(0, len(child_arch)-1)

            xop = random.randint(0, 3)
            while str(xop) == child_arch[node_id]:
                xop = random.randint(0, 3)
            child_arch[node_id] = xop
            #print(child_arch)
            child_arch = str(child_arch[0]) + "_" + str(child_arch[1]) + str(child_arch[2]) + "_" + \
                        str(child_arch[3]) + str(child_arch[4]) + str(child_arch[5])
            child_arch = arch.split("-")[0] + "-" + arch.split("-")[1] + "-" + child_arch
            return child_arch
        else: #macro
            raise Exception ('not implemented yet.')
    def get_arch_metrics(self, arch):
        # get arch metrics for all datasets/tasks
        arch = self.get_arch_str(arch)
        # get results for all tasks and interesting metrics
        results = {}
        all_results_unstructured = self.api.data[self.ss][arch].all_results
        #print(all_results_unstructured)
        for task in self.api.task_list:
            results[task] = {}
            #results[task]['model_info'] = all_results_unstructured[task]['model_info']
            for metric in self.important_metrics_per_task[task]:
                results[task][metric] = self.api.get_single_metric(arch, task, metric, mode='best')
        return results

class Nasbench201:
    def __init__(self, dataset, apiloc):
        self.dataset = dataset
        self.api = API(apiloc, verbose=False)
        self.epochs = '12'
    def get_config(self,uid, dataset):
        config = self.api.get_net_config(uid, 'cifar10-valid')
        return config
    def get_network(self, uid):
        #config = self.api.get_net_config(uid, self.dataset)
        config = self.api.get_net_config(uid, 'cifar10-valid')
        config['num_classes'] = 1
        network = get_cell_based_tiny_net(config)
        return network
    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_network(uid)
            yield uid, network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return 15625
    def num_activations(self):
        network = self.get_network(0)
        return network.classifier.in_features
    #def get_12epoch_accuracy(self, uid, acc_type, trainval, traincifar10=False):
    #    archinfo = self.api.query_meta_info_by_index(uid)
    #    if (self.dataset == 'cifar10' or traincifar10) and trainval:
    #        #return archinfo.get_metrics('cifar10-valid', acc_type, iepoch=12)['accuracy']
    #        return archinfo.get_metrics('cifar10-valid', 'x-valid', iepoch=12)['accuracy']
    #    elif traincifar10:
    #        return archinfo.get_metrics('cifar10', acc_type, iepoch=12)['accuracy']
    #    else:
    #        return archinfo.get_metrics(self.dataset, 'ori-test', iepoch=12)['accuracy']
    def get_12epoch_accuracy(self, uid, acc_type, trainval, traincifar10=False):
        #archinfo = self.api.query_meta_info_by_index(uid)
        if (self.dataset == 'cifar10' and trainval) or traincifar10:
            info = self.api.get_more_info(uid, 'cifar10-valid', iepoch=None, hp=self.epochs, is_random=True)
        else:
            info = self.api.get_more_info(uid, self.dataset, iepoch=None, hp=self.epochs, is_random=True)
        return info['valid-accuracy']
    def query_epoch(self, uid, epoch, args=None):
        info = self.api.get_more_info(int(uid), self.dataset, iepoch=epoch, is_random=True)
        try:
            return info['valid-accuracy']
        except:
            return info['test-accuracy']
    def get_final_accuracy(self, uid, acc_type, trainval):
        #archinfo = self.api.query_meta_info_by_index(uid)
        if self.dataset == 'cifar10' and trainval:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', 'x-valid')
            #info = self.api.query_by_index(uid, 'cifar10-valid', hp='200')
            #info = self.api.get_more_info(uid, 'cifar10-valid', iepoch=None, hp='200', is_random=True)
        else:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics(self.dataset, acc_type)
            #info = self.api.query_by_index(uid, self.dataset, hp='200')
            #info = self.api.get_more_info(uid, self.dataset, iepoch=None, hp='200', is_random=True)
        return info['accuracy']
        #return info['valid-accuracy']
        #if self.dataset == 'cifar10' and trainval:
        #    return archinfo.get_metrics('cifar10-valid', acc_type, iepoch=11)['accuracy']
        #else:
        #    #return archinfo.get_metrics(self.dataset, 'ori-test', iepoch=12)['accuracy']
        #    return archinfo.get_metrics(self.dataset, 'x-test', iepoch=11)['accuracy']
        ##dataset = self.dataset
        ##if self.dataset == 'cifar10' and trainval:
        ##    dataset = 'cifar10-valid'
        ##archinfo = self.api.get_more_info(uid, dataset, iepoch=None, use_12epochs_result=True, is_random=True)
        ##return archinfo['valid-accuracy']

    def get_accuracy(self, uid, acc_type, trainval=True):
        archinfo = self.api.query_meta_info_by_index(uid)
        if self.dataset == 'cifar10' and trainval:
            return archinfo.get_metrics('cifar10-valid', acc_type)['accuracy']
        else:
            return archinfo.get_metrics(self.dataset, acc_type)['accuracy']

    def get_accuracy_for_all_datasets(self, uid):
        archinfo = self.api.query_meta_info_by_index(uid,hp='200')

        c10 = archinfo.get_metrics('cifar10', 'ori-test')['accuracy']
        c10_val = archinfo.get_metrics('cifar10-valid', 'x-valid')['accuracy']

        c100 = archinfo.get_metrics('cifar100', 'x-test')['accuracy']
        c100_val = archinfo.get_metrics('cifar100', 'x-valid')['accuracy']

        imagenet = archinfo.get_metrics('ImageNet16-120', 'x-test')['accuracy']
        imagenet_val = archinfo.get_metrics('ImageNet16-120', 'x-valid')['accuracy']

        return c10, c10_val, c100, c100_val, imagenet, imagenet_val

    #def train_and_eval(self, arch, dataname, acc_type, trainval=True):
    #    unique_hash = self.__getitem__(arch)
    #    time = self.get_training_time(unique_hash)
    #    acc12 = self.get_12epoch_accuracy(unique_hash, acc_type, trainval)
    #    acc = self.get_final_accuracy(unique_hash, acc_type, trainval)
    #    return acc12, acc, time
    def train_and_eval(self, arch, dataname, acc_type, trainval=True, traincifar10=False):
        unique_hash = self.__getitem__(arch)
        time = self.get_training_time(unique_hash)
        acc12 = self.get_12epoch_accuracy(unique_hash, acc_type, trainval, traincifar10)
        acc = self.get_final_accuracy(unique_hash, acc_type, trainval)
        return acc12, acc, time
    def random_arch(self):
        return random.randint(0, len(self)-1)
    def get_training_time(self, unique_hash):
        #info = self.api.get_more_info(unique_hash, 'cifar10-valid' if self.dataset == 'cifar10' else self.dataset, iepoch=None, use_12epochs_result=True, is_random=True)
        #info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, use_12epochs_result=True, is_random=True)
        info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, hp='12', is_random=True)
        return info['train-all-time'] + info['valid-per-time']
        #if self.dataset == 'cifar10' and trainval:
        #    info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, hp=self.epochs, is_random=True)
        #else:
        #    info = self.api.get_more_info(unique_hash, self.dataset, iepoch=None, hp=self.epochs, is_random=True)

        ##info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, use_12epochs_result=True, is_random=True)
        #return info['train-all-time'] + info['valid-per-time']
    def mutate_arch(self, arch):
        op_names = get_search_spaces('cell', 'nas-bench-201')
        #config = self.api.get_net_config(arch, self.dataset)
        config = self.api.get_net_config(arch, 'cifar10-valid')
        parent_arch = Structure(self.api.str2lists(config['arch_str']))
        child_arch = deepcopy( parent_arch )
        node_id = random.randint(0, len(child_arch.nodes)-1)
        node_info = list( child_arch.nodes[node_id] )
        snode_id = random.randint(0, len(node_info)-1)
        xop = random.choice( op_names )
        while xop == node_info[snode_id][0]:
            xop = random.choice( op_names )
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple( node_info )
        arch_index = self.api.query_index_by_arch( child_arch )
        return arch_index




class Nasbench101:
    def __init__(self, dataset, apiloc, args):
        self.dataset = dataset
        self.api = nasbench101api.NASBench(apiloc)
        self.args = args
    def get_config(self, unique_hash):
        return self.get_spec(unique_hash)
    def get_accuracy(self, unique_hash, acc_type, trainval=True):
        spec = self.get_spec(unique_hash)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = 0.
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap['final_test_accuracy']
                if newacc > maxacc:
                    maxacc = newacc
        return maxacc*100
    def query_model(self, unique_hash, epochs=108, args=None):
        spec = self.get_spec(unique_hash)
        return self.api.query(spec, epochs)
        
    def query_epoch(self, unique_hash, epochs=12, args=None):
        spec = self.get_spec(unique_hash)
        #try:
        #    return spec['validation_accuracy']*100
        #except:
        #    return spec['test_accuracy']*100
        return self.api.query(spec, epochs)['validation_accuracy']
    def get_final_accuracy(self, uid, acc_type, trainval):
        return self.get_accuracy(uid, acc_type, trainval)*100
    def get_training_time(self, unique_hash):
        spec = self.get_spec(unique_hash)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = -1.
        maxtime = 0.
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap['final_test_accuracy']
                if newacc > maxacc:
                    maxacc = newacc
                    maxtime = statmap['final_training_time']
        return maxtime
    '''
    def get_network(self, idx: int):
        hash= self.__getitem__(idx)
        spec = self.get_spec(hash)
        network = Network(spec, self.args)
        return network
    '''
    def get_network(self, unique_hash):
        spec = self.get_spec(unique_hash)
        network = Network(spec, self.args)
        return network
    def get_spec(self, unique_hash):
        matrix = self.api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self.api.fixed_statistics[unique_hash]['module_operations']
        spec = ModelSpec(matrix, operations)
        return spec
    def __iter__(self):
        for unique_hash in self.api.hash_iterator():
            network = self.get_network(unique_hash)
            yield unique_hash, network
    def __getitem__(self, index):
        return next(itertools.islice(self.api.hash_iterator(), index, None))
    def __len__(self):
        return len(self.api.hash_iterator())
    def num_activations(self):
        for unique_hash in self.api.hash_iterator():
            network = self.get_network(unique_hash)
            return network.classifier.in_features
    def train_and_eval(self, arch, dataname, acc_type, trainval=True, traincifar10=False):
        #unique_hash = self.__getitem__(arch)
        unique_hash = arch
        time =12.* self.get_training_time(unique_hash)/108.
        acc = self.get_accuracy(unique_hash, acc_type, trainval)*100
        acc12=self.query_epoch(unique_hash, epochs=12, args=None)*100
        return acc12, acc, time
    def random_arch(self):
        return random.randint(0, len(self)-1)
        #return self.__getitem__(random.randint(0, len(self)-1))
    def mutate_arch(self, arch):
        #unique_hash = self.__getitem__(arch)
        unique_hash = arch
        matrix = self.api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self.api.fixed_statistics[unique_hash]['module_operations']
        coords = [ (i, j) for i in range(matrix.shape[0]) for j in range(i+1, matrix.shape[1])]
        random.shuffle(coords)
        # loop through changes until we find change thats allowed
        for i, j in coords:
            # try the ops in a particular order
            for k in [m for m in np.unique(matrix) if m != matrix[i, j]]:
                newmatrix = matrix.copy()
                newmatrix[i, j] = k
                spec = ModelSpec(newmatrix, operations)
                try:
                    newhash = self.api._hash_spec(spec)
                    if newhash in self.api.fixed_statistics:
                        return [n for n, m in enumerate(self.api.fixed_statistics.keys()) if m == newhash][0]
                except:
                    pass




class ReturnFeatureLayer(torch.nn.Module):
    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod
    def forward(self, x):
        return self.mod(x), x
                

def return_feature_layer(network, prefix=''):
    #for attr_str in dir(network):
    #    target_attr = getattr(network, attr_str)
    #    if isinstance(target_attr, torch.nn.Linear):
    #        setattr(network, attr_str, ReturnFeatureLayer(target_attr))
    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + '\t')


class NDS:
    def __init__(self, searchspace):
        self.searchspace = searchspace
        data = json.load(open(f'nds_data/{searchspace}.json', 'r'))
        try:
            data = data['top'] + data['mid']
        except Exception as e:
            pass
        self.data = data
    def __iter__(self):
        for unique_hash in range(len(self)):
            network = self.get_network(unique_hash)
            yield unique_hash, network
    def get_network_config(self, uid):
        return self.data[uid]['net']
    def get_network_optim_config(self, uid):
        return self.data[uid]['optim']
    def get_network(self, uid):
        netinfo = self.data[uid]
        config = netinfo['net']
        #print(config)
        if 'genotype' in config:
            #print('geno')
            gen = config['genotype']
            genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
            if '_in' in self.searchspace:
                network = NetworkImageNet(config['width'], 1, config['depth'], config['aux'],  genotype)
            else:
                network = NetworkCIFAR(config['width'], 1, config['depth'], config['aux'],  genotype)
            network.drop_path_prob = 0.
            #print(config)
            #print('genotype')
            L = config['depth']
        else:
            if 'bot_muls' in config and 'bms' not in config:
                config['bms'] = config['bot_muls']
                del config['bot_muls']
            if 'num_gs' in config and 'gws' not in config:
                config['gws'] = config['num_gs']
                del config['num_gs']
            config['nc'] = 1
            config['se_r'] = None
            config['stem_w'] = 12
            L = sum(config['ds'])
            if 'ResN' in self.searchspace:
                config['stem_type'] = 'res_stem_in'
            else:
                config['stem_type'] = 'simple_stem_in'
            #"res_stem_cifar": ResStemCifar,
            #"res_stem_in": ResStemIN,
            #"simple_stem_in": SimpleStemIN,
            if config['block_type'] == 'double_plain_block':
                config['block_type'] = 'vanilla_block'
            network = AnyNet(**config)
        return_feature_layer(network)
        return network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return len(self.data)
    def random_arch(self):
        return random.randint(0, len(self.data)-1)
    def query_epoch(self, uid, epoch=4, args=None):
        return 100.-self.data[uid]['test_ep_top1'][epoch-1]
    def get_final_accuracy(self, uid, acc_type, trainval):
        #print(self.data[uid])
        #quit()
        return 100.-self.data[uid]['test_ep_top1'][-1] #100- to transform to accuracy


def get_search_space(args):
    if args.nasspace == 'nasbench201':
        return Nasbench201(args.dataset, args.api_loc)
    elif args.nasspace == 'nasbench101':
        return Nasbench101(args.dataset, args.api_loc, args)
    elif args.nasspace == "transnas101":
        return Transnas101(args.dataset, args.api_loc, args)
    elif args.nasspace == 'nds_resnet':
        return NDS('ResNet')
    elif args.nasspace == 'nds_amoeba':
        return NDS('Amoeba')
    elif args.nasspace == 'nds_amoeba_in':
        return NDS('Amoeba_in')
    elif args.nasspace == 'nds_darts_in':
        return NDS('DARTS_in')
    elif args.nasspace == 'nds_darts':
        return NDS('DARTS')
    elif args.nasspace == 'nds_darts_fix-w-d':
        return NDS('DARTS_fix-w-d')
    elif args.nasspace == 'nds_darts_lr-wd':
        return NDS('DARTS_lr-wd')
    elif args.nasspace == 'nds_enas':
        return NDS('ENAS')
    elif args.nasspace == 'nds_enas_in':
        return NDS('ENAS_in')
    elif args.nasspace == 'nds_enas_fix-w-d':
        return NDS('ENAS_fix-w-d')
    elif args.nasspace == 'nds_pnas':
        return NDS('PNAS')
    elif args.nasspace == 'nds_pnas_fix-w-d':
        return NDS('PNAS_fix-w-d')
    elif args.nasspace == 'nds_pnas_in':
        return NDS('PNAS_in')
    elif args.nasspace == 'nds_nasnet':
        return NDS('NASNet')
    elif args.nasspace == 'nds_nasnet_in':
        return NDS('NASNet_in')
    elif args.nasspace == 'nds_resnext-a':
        return NDS('ResNeXt-A')
    elif args.nasspace == 'nds_resnext-a_in':
        return NDS('ResNeXt-A_in')
    elif args.nasspace == 'nds_resnext-b':
        return NDS('ResNeXt-B')
    elif args.nasspace == 'nds_resnext-b_in':
        return NDS('ResNeXt-B_in')
    elif args.nasspace == 'nds_vanilla':
        return NDS('Vanilla')
    elif args.nasspace == 'nds_vanilla_lr-wd':
        return NDS('Vanilla_lr-wd')
    elif args.nasspace == 'nds_vanilla_lr-wd_in':
        return NDS('Vanilla_lr-wd_in')