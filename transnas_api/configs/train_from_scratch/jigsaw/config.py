import os
import sys
import torch
from pathlib import Path

sys.path.insert(1, str((Path(__file__).parent / '..' / '..' / '..' / 'lib').resolve()))
from models.siamese import SiameseNet
from models.decoder import SiameseDecoder
from models.encoder import FFEncoder
import losses.all as loss_lib
import data.load_ops as load_ops
from optimizers.warmup import WarmupCosine


def get_cfg(encoder_str):
    cfg = {}

    # basics
    cfg['encoder_str'] = encoder_str
    cfg['config_dir'] = str(Path(__file__).parent.resolve())
    cfg['task_name'] = Path(cfg['config_dir']).name

    # paths
    cfg['root_dir'] = str((Path(__file__).parent / '..' / '..' / '..' / '..' / '..').resolve())
    cfg['dataset_dir'] = str(Path(cfg['root_dir']) / 'data/taskonomy_data/taskonomydata_mini')
    cfg['data_split_dir'] = str(Path(cfg['root_dir']) / 'tb101/code/experiments/final5k')
    cfg['log_dir'] = str(
        Path(cfg['root_dir']) / "tb101/benchmark_results/benchmark_results_local" / cfg['task_name'] / 'model_results' / cfg['encoder_str'])

    cfg['s3_dir'] = ''  # to setup in main.py
    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    # data loading
    cfg['batch_size'] = 128
    cfg['num_workers'] = 8

    # inputs
    cfg['input_dim'] = (255, 255)  # (1024, 1024)
    cfg['input_num_channels'] = 9

    # targets
    cfg['target_dim'] = 1000  # (1024, 1024)
    cfg['target_load_fn'] = load_ops.random_jigsaw_permutation
    cfg['target_load_kwargs'] = {
        'classes' : cfg['target_dim']
    }

    # demo
    cfg['demo_kwargs'] = {
        'classes': cfg['target_dim'],
        'mode': 'max'
    }

    #  transform
    # cfg['normal_params'] = {
    #     'mean': [0.5224, 0.5222, 0.5221],
    #     'std': [0.2234, 0.2235, 0.2236]
    # }
    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        # load_ops.ToTensor(),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64),
                                  centercrop=0.9, norm=False, totensor=True),
        # load_ops.Normalize(**cfg['normal_params'])
    ])
    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        # load_ops.ToTensor(),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64),
                                  centercrop=0.9, norm=False, totensor=True),
        # load_ops.Normalize(**cfg['normal_params'])
    ])

    # model
    cfg['encoder'] = FFEncoder(encoder_str, task_name=cfg['task_name']).network
    cfg['decoder_input_dim'] = (2048, 16, 16) if cfg['encoder_str'] == 'resnet50' else cfg['encoder'].output_dim
    cfg['decoder'] = SiameseDecoder(cfg['decoder_input_dim'], cfg['target_dim'], num_pieces=9)
    cfg['model_type'] = SiameseNet(cfg['encoder'], cfg['decoder'])

    # train
    cfg['fp16'] = False
    cfg['amp_opt_level'] = 'O1'
    cfg['num_epochs'] = 10
    cfg['criterion'] = loss_lib.CrossEntropyLoss()
    # cfg['optimizer'] = torch.optim.Adam
    cfg['optimizer'] = torch.optim.SGD
    cfg['initial_lr'] = 0.1
    cfg['lr_scheduler'] = WarmupCosine
    cfg['optimizer_kwargs'] = {
        'lr': cfg['initial_lr'],
        'momentum': 0.9,
        "nesterov": True,
        'weight_decay': 0.0005
    }

    # logging & model_db
    # cfg['identity_elems'] = {'seed': cfg['seed']}
    cfg['metric_content'] = ['train_top1', 'train_top5', 'train_loss', 'valid_top1', 'valid_top5', 'valid_loss',
                             'test_top1', 'test_top5', 'test_loss',
                             'time_elapsed']
    cfg['plot_msg'] = ''

    return cfg
