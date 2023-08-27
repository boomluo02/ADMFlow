import os,sys
prjt_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prjt_path)
import json
import argparse
import torch
import torch.nn as nn
import utils.helper_functions as helper

from torch.utils.data.dataloader import DataLoader

from model.admflow import ADMFlow
from loader.MDR import MDREventFlow

from utils.test_utils import TestRaftEvents

def test(args):
    
    config_path = 'config/mdr.json'
    config = json.load(open(os.path.join(prjt_path, config_path)))

    config["data_loader"]["test"]["args"].update({"sequence": args.test_sequence, "event_interval": args.event_interval})
    config['name'] = os.path.join(config['name'], "test_mdr_{:s}".format(args.event_interval))

    if config['cuda'] and not torch.cuda.is_available():
        print('Warning: There\'s no CUDA support on this machine')
    
    if(args.eval_type):
        config["data_loader"]["test"]["args"].update({"eval_type": "sparse"}) #测试时屏蔽没有event点的数据

    # Create Save Folder
    # config.update({"without_res":"no_res"})
    save_path = helper.create_save_path(os.path.join(prjt_path, config['save_dir']), config['name'].lower(), restart=True)

    # Copy config file to save dir
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'),
              indent=4, sort_keys=False)
    # Logger

    test_logger = helper.Logger(save_path, custom_name='test.log')
    test_logger.initialize_file("test")

    test_set = MDREventFlow(
        args = config["data_loader"]["test"]["args"],
        train=False
    )

    # Instantiate Dataloader
    test_set_loader = DataLoader(test_set,
                                 batch_size=config['data_loader']['test']['args']['batch_size'],
                                 shuffle=config['data_loader']['test']['args']['shuffle'],
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=True)
    
    # Load Model
    model = ADMFlow(
        config=config, 
        n_first_channels=config['data_loader']['train']['args']['num_voxel_bins']
    )

    states = torch.load(os.path.join(prjt_path, "checkpoint/MDR_{:s}.pth.tar".format(args.event_interval)))

    state_dict = {}
    for key, param in states['state_dict'].items():
        state_dict.update({key.replace('module.',''): param})
    model.load_state_dict(state_dict)
    args.best_epe = states['epe']
    start_epoch = states['epoch']


    additional_loader_returns = None
    test = TestRaftEvents(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=test_logger,
        save_path=save_path,
        additional_args=additional_loader_returns
    )

    model = nn.DataParallel(model)

    mepe = test.test_multi_sequence(model, start_epoch + 1, sequence_list=["0.09_0.24", "0.24_0.39", "0.39_0.54", "0.54_0.69"], stride=1, visualize_map=True, vis_events=True)
    test_logger.write_line('=== Current epoch {:d} val best epe: {:.6f} ==='.format(start_epoch + 1, mepe))
    return
            
if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-be','--best_epe', default=1e5, type=float)
    parser.add_argument('--test_sequence', '-sq', default='0.09_0.24', type=str)
    parser.add_argument('--event_interval', '-dt', choices=['dt1','dt4'], default=None, type=str, help='Input setting of events interval')
    parser.add_argument('--eval_type', '-eval',action='store_true')
    args = parser.parse_args()

    # Run Test Script
    test(args)
