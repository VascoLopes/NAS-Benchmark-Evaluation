from api import TransNASBenchAPI
import utils
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--cfg_dir', type=str, default='./configs/train_from_scratch/class_object/', help='directory containing config.py file')
    parser.add_argument('--encoder_str', type=str, default='resnet50')
    parser.add_argument('--nopause', dest='nopause', action='store_true')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--ddp', action='store_true')
    parser.set_defaults(nopause=True)
    args = parser.parse_args()
    device_list = list(range(torch.cuda.device_count()))

    api = TransNASBenchAPI("../datasets/transnas-bench_v10141024.pth")

    #cfg = utils.setup_config(args, len(device_list))
    #rank = 0
    #print(cfg)
    #model = utils.setup_model(cfg, device_list, rank, ddp=False)
    #print(model)

    '''
    df=open('arch2','w')
    for network in api:
        print(network)
        df.write(str(network))
        df.write('\n')
    df.close()
    quit()
    '''

    print ( api.task_list)
    
    #xarch = '64-310234-basic'
    xarch = '64-231313-basic'
    for xtask in api.task_list:
        print(f'----- {xtask} -----')
        print(f'--- info ---')
        #for xinfo in api.info_names:
        #    print(f"{xinfo} : {api.get_model_info(xarch, xtask, xinfo)}")
        print(f'--- metrics ---')
        print(api.metrics_dict[xtask])
        
        for xmetric in api.metrics_dict[xtask]:
            print(f"{xmetric} : {api.get_single_metric(xarch, xtask, xmetric, mode='best')}")
            #print(f"best epoch : {api.get_best_epoch_status(xarch, xtask, metric=xmetric)}")
        # print(f"final epoch : {api.get_epoch_status(xarch, xtask, epoch=-1)}")
            if ('valid' in xmetric and 'loss' not in xmetric) or ('valid' in xmetric and 'neg_loss' in xmetric):
                print(f"\nbest_arch -- {xmetric}: {api.get_best_archs(xtask, xmetric, 'micro')[0]}")
        
