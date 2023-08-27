import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)
import numpy as np
import torch
import cv2
import json
import h5py
import pandas
import glob

from matplotlib import pyplot as plt
from matplotlib import colors
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from loader_utils import get_compressed_events, read_flo
from loader_utils import DenseSparseAugmentor, EventSequence, EventSequenceToVoxelGrid_Pytorch

import pdb


class MDREventFlow(Dataset):
    def __init__(self, args, train = True):
        super(MDREventFlow, self)

        self.input_type = 'events'
        self.type = 'train' if train else 'val'
        self.evaluation_type = args['eval_type']
        self.dt = args['event_interval']

        if(self.type == 'train'):
            self.get_train_sequence("batch_1")
        elif(self.type == 'val'):
            self.change_test_sequence(args['sequence'])

        self.image_width = 346
        self.image_height = 260
        self.num_bins = args['num_voxel_bins']
        self.voxel = EventSequenceToVoxelGrid_Pytorch(
            num_bins=self.num_bins, 
            normalize=True, 
            gpu=True
        )
        self.cropper = transforms.CenterCrop((256,256))
        
        if 'aug_params' in args.keys():
            self.aug_params = args['aug_params']
            self.dense_augmentor = DenseSparseAugmentor(**self.aug_params)
        else:
            self.augmentor = None

    def get_train_sequence(self, sequence):
        self.sequence = sequence

        self.events1_path = os.path.join(proc_path, 'dataset/MDR/dt1/train/{:s}/events1'.format(self.sequence))
        self.events2_path = os.path.join(proc_path, 'dataset/MDR/dt1/train/{:s}/events2'.format(self.sequence))
        self.d_events1_path = os.path.join(proc_path, 'dataset/MDR/dt1/train/{:s}/best_density_events1'.format(self.sequence))
        self.d_events2_path = os.path.join(proc_path, 'dataset/MDR/dt1/train/{:s}/best_density_events2'.format(self.sequence))
        self.flow_path = os.path.join(proc_path, 'dataset/MDR/dt1/train/{:s}/flow'.format(self.sequence))
        
        self.names = []

        self.events1_list = []
        self.events2_list = []
        self.d_events1_list = []
        self.d_events2_list = []
        self.flow_list = []

        for root, dirs, files in os.walk(self.events1_path):

            for event_file in files:
                if event_file.endswith('.npz'):
                    
                    name = root.rsplit("/", 1)[-1]
                    event1_file_path = os.path.join(self.events1_path, name, event_file)
                    event2_file_path = os.path.join(self.events2_path, name, event_file.replace("events1", "events2"))
                    d_event1_file_path = os.path.join(self.d_events1_path, "{:s}_best_density_events1.npz".format(name))
                    d_event2_file_path = os.path.join(self.d_events2_path, "{:s}_best_density_events2.npz".format(name))
                    flow_file_path = os.path.join(self.flow_path, "{:s}_flow.flo".format(name))


                    if(os.path.exists(event2_file_path) and os.path.exists(flow_file_path) and \
                        os.path.exists(d_event1_file_path) and os.path.exists(d_event2_file_path)):

                        self.names.append(event_file.replace(".npz", "").replace("events1", ""))
                        self.events1_list.append(event1_file_path)
                        self.events2_list.append(event2_file_path)
                        self.d_events1_list.append(d_event1_file_path)
                        self.d_events2_list.append(d_event2_file_path)
                        self.flow_list.append(flow_file_path)


    def change_test_sequence(self, sequence):
        self.sequence = sequence
        # pdb.set_trace()
        self.events1_path = os.path.join(proc_path, 'dataset/MDR/{:s}/test/{:s}/events1'.format(self.dt, self.sequence))
        self.events2_path = os.path.join(proc_path, 'dataset/MDR/{:s}/test/{:s}/events2'.format(self.dt, self.sequence))
        self.flow_path = os.path.join(proc_path, 'dataset/MDR/{:s}/test/{:s}/flow'.format(self.dt, self.sequence))
        
        self.names = []

        self.events1_list = []
        self.events2_list = []
        self.flow_list = []

        for root, dirs, files in os.walk(self.events1_path):

            for event_file in files:
                if event_file.endswith('.npz'):
                    
                    name = root.rsplit("/", 1)[-1]
                    event1_file_path = os.path.join(self.events1_path, name, event_file)
                    event2_file_path = os.path.join(self.events2_path, name, event_file.replace("events1", "events2"))
                    flow_file_path = os.path.join(self.flow_path, "{:s}_flow.flo".format(name))


                    if(os.path.exists(event2_file_path) and os.path.exists(flow_file_path)):

                        self.names.append(event_file.replace(".npz", "").replace("events1", ""))
                        self.events1_list.append(event1_file_path)
                        self.events2_list.append(event2_file_path)
                        self.flow_list.append(flow_file_path)


    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__ + " for {}".format(self.type), True)

    def get_sample(self, idx):

        names = self.names[idx]

        # Load Flow
        flow = read_flo(self.flow_list[idx])

        if(flow.shape[-1]==2):
            flow = flow.transpose(2,0,1)

        return_dict = {'idx': names,
                    'flow': torch.from_numpy(flow),
                    "valid": None
                    }

        # Load Events 
        params = {'height': self.image_height, 'width': self.image_width}
        
        event_path_old = self.events1_list[idx]
        event_path_new = self.events2_list[idx]
        
        events_old = get_compressed_events(event_path_old)
        events_new = get_compressed_events(event_path_new)
        
        ev_seq_old = EventSequence(None, params, features=events_old, timestamp_multiplier=1e6, convert_to_relative=True)
        ev_seq_new = EventSequence(None, params, features=events_new, timestamp_multiplier=1e6, convert_to_relative=True)
        event_volume_old = self.voxel(ev_seq_old).cpu()
        event_volume_new = self.voxel(ev_seq_new).cpu()

        return_dict['event_volume_new'] = event_volume_new
        return_dict['event_volume_old'] = event_volume_old 

        if(self.type == 'train'):

            d_event_path_old = self.d_events1_list[idx]
            d_event_path_new = self.d_events2_list[idx]
            
            d_events_old = get_compressed_events(d_event_path_old)
            d_events_new = get_compressed_events(d_event_path_new)
            
            d_ev_seq_old = EventSequence(None, params, params, features=d_events_old, timestamp_multiplier=1e6, convert_to_relative=True)
            d_ev_seq_new = EventSequence(None, params, params, features=d_events_new, timestamp_multiplier=1e6, convert_to_relative=True)
            
            d_event_volume_old = self.voxel(d_ev_seq_old).cpu()
            d_event_volume_new = self.voxel(d_ev_seq_new).cpu()
        
            return_dict['d_event_volume_old'] = d_event_volume_old 
            return_dict['d_event_volume_new'] = d_event_volume_new

        elif(self.type == 'val'):

            seq = ev_seq_old.get_sequence_only()
            h = self.image_height
            w = self.image_width
            hist, _, _ = np.histogram2d(x=seq[:,1], y=seq[:,2],
                                    bins=(w,h),
                                    range=[[0,w], [0,h]])
            hist = hist.transpose()
            ev_mask = hist > 0
            return_dict['event_valid'] = torch.from_numpy(ev_mask).unsqueeze(dim=0)

        return return_dict


    def __len__(self):
        
        return len(self.names)

    def __getitem__(self, idx):
        
        sample = self.get_sample(idx % len(self))

        if self.type == 'train':
            
            event1 = sample['event_volume_old'].permute(1,2,0).numpy()
            event2 = sample['event_volume_new'].permute(1,2,0).numpy()
            flow = sample['flow'].permute(1,2,0).numpy()

            if("d_event_volume_old" in sample.keys()):
            
                d_event1 = sample['d_event_volume_old'].permute(1,2,0).numpy()
                d_event2 = sample['d_event_volume_new'].permute(1,2,0).numpy()
                
                event1, event2, d_event1, d_event2, flow_crop = self.dense_augmentor(event1, event2, d_event1, d_event2, flow)

                valid = np.logical_and(np.logical_and(~np.isinf(flow_crop[:, :, 0]), ~np.isinf(flow_crop[:, :, 1])), np.linalg.norm(flow_crop, axis=2) > 0)

                sample['event_volume_old'] = torch.from_numpy(event1).permute(2, 0, 1).float()
                sample['event_volume_new'] = torch.from_numpy(event2).permute(2, 0, 1).float()
                sample['d_event_volume_old'] = torch.from_numpy(d_event1).permute(2, 0, 1).float()
                sample['d_event_volume_new'] = torch.from_numpy(d_event2).permute(2, 0, 1).float()
                sample['flow'] = torch.from_numpy(flow_crop).permute(2, 0, 1).float()
                sample['valid'] = torch.from_numpy(valid).float()
            
            else:
                
                event1, event2, flow = self.augmentor(event1, event2, flow)
                valid = np.logical_and(np.logical_and(~np.isinf(flow[:, :, 0]), ~np.isinf(flow[:, :, 1])), np.linalg.norm(flow, axis=2) > 0)

                sample['event_volume_old'] = torch.from_numpy(event1).permute(2, 0, 1).float()
                sample['event_volume_new'] = torch.from_numpy(event2).permute(2, 0, 1).float()
                sample['flow'] = torch.from_numpy(flow).permute(2, 0, 1).float()
                sample['valid'] = torch.from_numpy(valid).float()

        elif self.type == 'val':

            sample['flow'] = self.cropper(sample['flow'])
            sample['valid'] = (sample['flow'][0].abs() < 1000) & (sample['flow'][1].abs() < 1000)
            sample['event_volume_new'] = self.cropper(sample['event_volume_new'])
            sample['event_volume_old'] = self.cropper(sample['event_volume_old'])
            sample['event_valid'] = self.cropper(sample['event_valid'])

        return sample

if __name__ == '__main__':
    config_path = os.path.join(proc_path, 'config/mdr.json')
    config = json.load(open(config_path))

    config["data_loader"]["test"]["args"].update({"sequence": "0.09_0.24"})

    test_set = MDREventFlow(
        args = config["data_loader"]["test"]["args"],
        train=False
    )

    test_set_loader = DataLoader(test_set,
                            batch_size=2,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)

    for batch_idx, data in enumerate(test_set_loader):

        idx = data['idx']
        event_volume_old = data['event_volume_old']
        flow = data['flow']
        print(event_volume_old.max())