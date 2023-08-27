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

from utils.train_utils import TrainDenseSparse
from utils.test_utils import TestRaftEvents

def train(args):
    
    config_path = 'config/mvsec.json'
    config = json.load(open(os.path.join(prjt_path, config_path)))

    config["train"]["lr"] = args.lr
    config["train"]["wdecay"] = args.wd
    config["train"]["num_steps"] =  args.train_iters
    config['data_loader']['train']['args']['batch_size'] = args.batch_size

    config["data_loader"]["train"]["args"].update({"sequence": "outdoor_day1"})
    config["data_loader"]["test"]["args"].update({"sequence": args.test_sequence})
    config['name'] = os.path.join(config['name'], "{:s}_lr{:5f}_we{:5f}".format(args.event_interval, args.lr, args.wd))

    if config['cuda'] and not torch.cuda.is_available():
        print('Warning: There\'s no CUDA support on this machine, '
                            'training is performed on CPU.')
    
    if(args.eval_type):
        config["data_loader"]["test"]["args"].update({"eval_type": "sparse"}) #测试时屏蔽没有event点的数据

    # Create Save Folder
    config.update({"without_res":"no_res"})
    save_path = helper.create_save_path(os.path.join(prjt_path, config['save_dir']), config['name'].lower(), restart=(args.start_epoch))

    # Copy config file to save dir
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'),
              indent=4, sort_keys=False)
    # Logger
    train_logger = helper.Logger(save_path, custom_name='train.log')
    train_logger.initialize_file("train")

    test_logger = helper.Logger(save_path, custom_name='test.log')
    test_logger.initialize_file("test")

    if(args.event_interval == "dt1"):
        from loader.MVSEC import MvsecEventFlow
        train_set = MvsecEventFlow(
            args = config["data_loader"]["train"]["args"],
            train=True
        )
        test_set = MvsecEventFlow(
            args = config["data_loader"]["test"]["args"],
            train=False
        )
    elif(args.event_interval == "dt4"):
        from loader.MVSEC import MvsecEventFlow_dt4
        train_set = MvsecEventFlow_dt4(
            args = config["data_loader"]["train"]["args"],
            train=True
        )
        test_set = MvsecEventFlow_dt4(
            args = config["data_loader"]["test"]["args"],
            train=False
        )
    else:
         raise Exception('Please provide a valid input setting (dt1 or dt4)!')


    # Instantiate Dataloader
    train_set_loader = DataLoader(train_set,
                                batch_size=config['data_loader']['train']['args']['batch_size'],
                                shuffle=config['data_loader']['train']['args']['shuffle'],
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=True)
    test_set_loader = DataLoader(test_set,
                                 batch_size=config['data_loader']['test']['args']['batch_size'],
                                 shuffle=config['data_loader']['test']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=True)
    
    # Load Model
    model = ADMFlow(
        config=config, 
        n_first_channels=config['data_loader']['train']['args']['num_voxel_bins']
    )

    if args.start_epoch:

        states = torch.load(os.path.join(prjt_path, "checkpoint/mvsec_{:s}.pth.tar".format(args.event_interval)))

        state_dict = {}
        for key, param in states['state_dict'].items():
            state_dict.update({key.replace('module.',''): param})
        model.load_state_dict(state_dict)
        args.best_epe = states['epe']
        start_epoch = states['epoch']
    else:
        start_epoch = 0

    additional_loader_returns = None
    train = TrainDenseSparse(
        model=model,
        config=config,
        args=args,
        data_loader=train_set_loader,
        train_logger=train_logger,
        save_path=save_path,
        additional_args=additional_loader_returns,
        visualizer_map=True
    )
    train.fetch_optimizer(model)

    test = TestRaftEvents(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=test_logger,
        save_path=save_path,
        additional_args=additional_loader_returns
    )

    model = nn.DataParallel(model)

    # test.summary()
    if args.test_only:
        # test.summary()
        # mepe = test(model, start_epoch + 1)
        mepe = test.test_multi_sequence(model, start_epoch + 1, sequence_list=['outdoor_day1','indoor_flying1','indoor_flying2','indoor_flying3'], stride=1, visualize_map=True, vis_events=True)
        test_logger.write_line('=== Current epoch {:d} val best epe: {:.6f} ==='.format(start_epoch + 1, mepe))
        return

    epochs = args.train_iters // args.val_iters

    for epoch in range(start_epoch, epochs):

        train.summary()
        model = train.train_mimounet_iters(model, start_epoch=epoch, val_iters=args.val_iters, compute_density=True)

        save_dict = {'epoch': epoch,
            'state_dict': model.state_dict()}
        # torch.save(save_dict, os.path.join(save_path, 'train_lasted_ckpt.pth.tar'))

        test.summary()
        # mepe = test(model, epoch)
        mepe = test.test_multi_sequence(model, epoch, ['outdoor_day1','indoor_flying1','indoor_flying2','indoor_flying3'])

        save_dict = {'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'epe': mepe}

        torch.save(save_dict, os.path.join(save_path, 'lasted_ckpt.pth.tar'))
        
        if mepe <= args.best_epe:
            args.best_epe = mepe
            
            if os.path.exists(os.path.join(save_path, 'best_epe.pth.tar')):
                os.remove(os.path.join(save_path, 'best_epe.pth.tar'))
            if not os.path.exists(os.path.join(save_path, 'best_epe.pth.tar')):
                shutil.copyfile(os.path.join(save_path,'lasted_ckpt.pth.tar'), os.path.join(save_path,'best_epe.pth.tar'))
        test_logger.write_line('=== Current val best epe: {:.6f} ==='.format(args.best_epe))
            
if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_workers', default=0, type=int, help='How many sub-processes to use for data loading')
    parser.add_argument('--train_iters', default=5000000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-se','--start-epoch', action='store_true', help='restart')
    parser.add_argument('-be','--best_epe', default=1e5, type=float)
    parser.add_argument('--val_iters', default=1000, type=int, metavar='N', help='Evaluate every \'evaluate interval')
    parser.add_argument('--lr', default=5e-4, type=float, help='learnning rate')
    parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')

    parser.add_argument('--batch_size', '-bs', default=6, type=int, help='batch size in trainning')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_sequence', '-sq', default='indoor_flying2', type=str)
    parser.add_argument('--event_interval', '-dt', choices=['dt1','dt4'], default=None, type=str, help='Input setting of events interval')
    parser.add_argument('--eval_type', '-eval',action='store_true')
    args = parser.parse_args()

    # Run Test Script
    train(args)
