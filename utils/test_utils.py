import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
import numpy as np
import pandas as pd
import torch
from torchvision import utils
import helper_functions as helper
from helper_functions import *

import cv2

from matplotlib import colors
from utils.tools import tensor_tools

import pdb

class Test(object):
    """
    Test class

    """

    def __init__(self, model, config, data_loader, 
                test_logger=None, save_path=None, additional_args=None, visualizer_map=True, normal=False, save_excel=False):

        self.downsample = False # Downsampling for Rebuttal
        self.config = config
        self.data_loader = data_loader
        self.additional_args = additional_args

        self.model = model
        self.gpu = torch.device('cuda')
        
        if save_path is None:
            self.save_path = helper.create_save_path(config['save_dir'].lower(),
                                           config['name'].lower())
        else:
            self.save_path=save_path
        
        if 'flow_direction' in config.keys():
            if config['flow_direction'] == 'forward':
                self.flow_save_path = os.path.join('/home/luoxinglong/workspace/E-RAFT/dataset/outdoor_day2', 'flow_fw')
                if not os.path.exists(self.flow_save_path):
                    os.makedirs(self.flow_save_path)
            elif config['flow_direction'] == 'backward':
                self.flow_save_path = os.path.join('/home/luoxinglong/workspace/E-RAFT/dataset/outdoor_day2', 'flow_bw')
                if not os.path.exists(self.flow_save_path):
                    os.makedirs(self.flow_save_path)

        self.image_size = config["val_img_size"]

        if test_logger is None:
            self.logger = helper.Logger(self.save_path, "test_log.txt")
        else:
            self.logger = test_logger

        self.visualize_map = visualizer_map
        self.normal = normal

        self.print_iter = 1
        self.is_car = False
        if(("sequence" in config["data_loader"]["test"]["args"].keys())):
            if("outdoor" in config["data_loader"]["test"]["args"]["sequence"]):
                self.is_car = True
                self.logger.write_line("!!!is_car == True!!!", True)
                self.print_iter = 1
            elif("indoor" in config["data_loader"]["test"]["args"]["sequence"]):
                self.logger.write_line("!!!is_car == False!!!", True)
                self.print_iter = 100

        self.save_excel = save_excel
        if(save_excel):
            self.excel = []
        
        self.logger.write_line("evaluation type is {:s}".format(self.data_loader.dataset.evaluation_type), True)

    def summary(self):
        self.logger.write_line("====================================== TEST SUMMARY ======================================", True)
        self.logger.write_line("Model:\t\t\t" + self.model.__class__.__name__, True)
        self.logger.write_line("Tester:\t\t" + self.__class__.__name__, True)
        self.logger.write_line("Test Set:\t" + self.data_loader.dataset.__class__.__name__, True)
        self.logger.write_line("\t-Dataset length:\t"+str(len(self.data_loader)), True)
        self.logger.write_line("\t-Batch size:\t\t" + str(self.data_loader.batch_size), True)
        self.logger.write_line("==========================================================================================", True)

    def __call__(self, model, epoch):
        mepe = self._test(model, epoch)
        return mepe

    def run_network(self, model, batch):
        raise NotImplementedError

    def get_input_events(self, batch):
        raise NotImplementedError
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'valid' in batch.keys():
                return batch['flow_est'].cpu().data, (batch['flow'].cpu().data, batch['valid'].cpu().data)
            return batch['flow_est'].cpu().data, batch['flow'].cpu().data
        else:
            f_est = batch['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch['flow'].cpu().data, scale_factor=0.5)
            if 'valid' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['valid'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_events(self, batch):
        if not self.downsample:
            events0 = batch['event_volume_old'].cpu().data
            events1 = batch['event_volume_new'].cpu().data
        else:
            events0 = torch.nn.functional.interpolate(batch['event_volume_old'].cpu().data, scale_factor=0.5)
            events1 = torch.nn.functional.interpolate(batch['event_volume_new'].cpu().data, scale_factor=0.5)
        return events0, events1
    
    def get_dense_events(self, batch):
        if not self.downsample:
            events0 = batch['d_event_volume_old'].cpu().data
            events1 = batch['d_event_volume_new'].cpu().data
        else:
            events0 = torch.nn.functional.interpolate(batch['d_event_volume_old'].cpu().data, scale_factor=0.5)
            events1 = torch.nn.functional.interpolate(batch['d_event_volume_new'].cpu().data, scale_factor=0.5)
        return events0, events1

    def check_tensor(self, data, name, print_data=False, print_in_txt=None):
        if data.is_cuda:
            temp = data.detach().cpu().numpy()
        else:
            temp = data.detach().numpy()
        a = len(name)
        name_ = name + ' ' * 100
        name_ = name_[0:max(a, 10)]
        print_str = '%s, %s, %s, %s,%s,%s,%s,%s' % (name_, temp.shape, data.dtype, ' max:%.2f' % np.max(temp), ' min:%.2f' % np.min(temp),
                                                    ' mean:%.2f' % np.mean(temp), ' sum:%.2f' % np.sum(temp), data.device)
        if print_in_txt is None:
            print(print_str)
        else:
            print(print_str, file=print_in_txt)
        if print_data:
            print(temp)
        return print_str

    def vis_map(self, map, name):
        map = map[0].numpy()
        map_img = np.squeeze(np.sum(map, axis=0))
        map_img = (map_img - map_img.min()) / (map_img.max() - map_img.min())
        map_img = np.asarray(map_img * 255, dtype=np.uint8)
        save_map_path = os.path.join(self.save_path, 'test')
        if not os.path.exists(save_map_path):
            os.makedirs(save_map_path)
        cv2.imwrite(os.path.join(save_map_path, name), map_img)
        return 
    
    def vis_mask(self, mask, name):
        mask = mask[0].numpy()
        map_img = np.squeeze(np.sum(mask, axis=0))
        map_img = np.asarray(map_img * 255, dtype=np.uint8)
        save_map_path = os.path.join(self.save_path, 'test')
        if not os.path.exists(save_map_path):
            os.makedirs(save_map_path)
        cv2.imwrite(os.path.join(save_map_path, name), map_img)
        return 
    

    def vis_map_RGB(self, map, name):

        def plot_points_on_background(points_coordinates,
                              background,
                              points_color=[0, 0, 255]):
            """
            Args:
                points_coordinates: array of (y, x) points coordinates
                                    of size (number_of_points x 2).
                background: (3 x height x width)
                            gray or color image uint8.
                color: color of points [red, green, blue] uint8.
            """
            if not (len(background.size()) == 3 and background.size(0) == 3):
                raise ValueError('background should be (color x height x width).')
            _, height, width = background.size()
            background_with_points = background.clone()
            y, x = points_coordinates.transpose(0, 1)
            if len(x) > 0 and len(y) > 0: # There can be empty arrays!
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
                    raise ValueError('points coordinates are outsize of "background" '
                                    'boundaries.')
                background_with_points[:, y, x] = torch.Tensor(points_color).type_as(
                    background).unsqueeze(-1)
            return background_with_points 
        
        def get_map_img_sum(map):

            map = map[0].numpy()
            _,height,width = map.shape

            map_img = np.squeeze(np.sum(map, axis=0))
            # if(map_img.mean() != 0):
            #     map_img = (map_img - map_img.min()) / (map_img.max() - map_img.min())
            density = np.sum(np.abs(map_img)>0.5) / (height * width)

            mean = map_img.mean()

            # Red -> Negative Events
            red = (map_img <= (mean-0.2))
            # Blue -> Positive Events
            blue = (map_img >= (mean+0.2))

            # # Red -> Negative Events
            # red = (map_img <= (mean-0.9))
            # # Blue -> Positive Events
            # blue = (map_img >= (mean+0.9))

            background = torch.full((3, height, width), 255).byte()
            points_on_background = plot_points_on_background(
            torch.nonzero(torch.from_numpy(red.astype(np.uint8))), background,
            [255, 0, 0])
            points_on_background = plot_points_on_background(
            torch.nonzero(torch.from_numpy(blue.astype(np.uint8))),
            points_on_background, [0, 0, 255])

            map_img_sum = points_on_background.permute(1,2,0).numpy()
            return map_img_sum, density

        save_map_path = os.path.join(self.save_path, 'test')
        os.makedirs(save_map_path, exist_ok=True)
        name = name.replace(".jpg","")

        if(isinstance(map, dict)):
            for key, map1 in map.items():
                map_img_sum, density = get_map_img_sum(map1)
                cv2.imwrite(os.path.join(save_map_path, name.replace(".jpg", "")+"_"+str(key)+"_{:.3f}.jpg".format(density)), map_img_sum)
        else:
            map_img_sum, density = get_map_img_sum(map)
            cv2.imwrite(os.path.join(save_map_path, name+"_{:.3f}.jpg".format(density)), map_img_sum)

        return 
    
    def compute_map_density(self, map):
        map = map[0].numpy()
        channel,h,w = map.shape
        sum_map = np.zeros((h,w))
        # pdb.set_trace()
        for c in range(channel):
            sum_map = sum_map + np.abs(map[c])

        density = np.sum(sum_map>0.5) / (h * w)

        return density

    def visualize_optical_flow_light(self, flow, name=None):
        # out = flow_to_image(flow*100)
        out = tensor_tools.flow_to_image_dmax((flow*100)**2)
        save_flow_path = os.path.join(self.save_path, 'test')
        if not os.path.exists(save_flow_path):
            os.makedirs(save_flow_path)
        cv2.imwrite(os.path.join(save_flow_path, name), out)
        return

    def visualize_optical_flow(self, flow, name=None, events_mask=None):
        # out = flow_to_image(flow)
            # flow = flow.transpose(1,2,0)
        # flow = (flow * 1000)**5
        if(flow.shape[0]==2):
            flow = flow.transpose(1,2,0)
        flow[np.isinf(flow)]=0
        # Use Hue, Saturation, Value colour model
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

        # The additional **0.5 is a scaling factor
        mag = np.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

        ang = np.arctan2(flow[...,1], flow[...,0])
        ang[ang<0]+=np.pi*2
        hsv[..., 0] = ang/np.pi/2.0 # Scale from 0..1
        hsv[..., 1] = 1
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
        rgb = colors.hsv_to_rgb(hsv)
        # This all seems like an overkill, but it's just to exactly match the cv2 implementation
        bgr = np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)
        out = bgr*255

        save_flow_path = os.path.join(self.save_path,'test')
        os.makedirs(save_flow_path, exist_ok=True)

        if(events_mask is not None):
            out_masked = out*events_mask
            cv2.imwrite(os.path.join(save_flow_path, name.replace("flow","flow_masked")), out_masked)

        cv2.imwrite(os.path.join(save_flow_path, name), out)

        return


    def flow_error(self, flow_gt, flow_pred, event_img, is_car=True):
        
        flow_gt = flow_gt[0].numpy().transpose(1,2,0)
        flow_pred = flow_pred[0].numpy().transpose(1,2,0)
        event_img = event_img.numpy()
        
        max_row = flow_gt.shape[1]
        if is_car == True:
            max_row = 190
        flow_gt_cropped = flow_gt[:max_row, :]
        flow_pred_cropped = flow_pred[:max_row, :]

        flow_mask = np.logical_and(np.logical_and(~np.isinf(flow_gt_cropped[:, :, 0]), ~np.isinf(flow_gt_cropped[:, :, 1])), np.linalg.norm(flow_gt_cropped, axis=2) > 0)
        
        if self.data_loader.dataset.evaluation_type == "sparse":
            event_img_cropped = np.squeeze(event_img)[:max_row, :]
            event_mask = event_img_cropped > 0
            total_mask = np.squeeze(np.logical_and(event_mask, flow_mask))
            gt_masked = flow_gt_cropped[total_mask, :]
            pred_masked = flow_pred_cropped[total_mask, :]
        elif self.data_loader.dataset.evaluation_type == "dense":
            gt_masked = flow_gt_cropped[flow_mask, :]
            pred_masked = flow_pred_cropped[flow_mask, :]

        # gt_masked = flow_gt_cropped[flow_mask, :]
        # pred_masked = flow_pred_cropped[flow_mask, :]

        EE = np.linalg.norm(gt_masked - pred_masked, axis=-1)
        EE_gt = np.linalg.norm(gt_masked, axis=-1)

        n_points = EE.shape[0]

        thresh = 1.
        percent_1_AEE = float((EE < thresh).sum()) / float(EE.shape[0] + 1e-5)

        thresh = 3.
        percent_3_AEE = float((EE < thresh).sum()) / float(EE.shape[0] + 1e-5)

        EE = torch.from_numpy(EE)
        EE_gt = torch.from_numpy(EE_gt)

        if torch.sum(EE) == 0:
            AEE = 0
            AEE_sum_temp = 0

            AEE_gt = 0
            AEE_sum_temp_gt = 0
        else:
            AEE = torch.mean(EE)
            AEE_sum_temp = torch.sum(EE)

            AEE_gt = torch.mean(EE_gt)
            AEE_sum_temp_gt = torch.sum(EE_gt)

        return AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt


    def _test(self, model, epoch=0):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.eval()
        try:
            model.module.freeze_bn()
        except:
            pass
        
        AEE_sum = 0.
        AEE_sum_sum = 0.
        AEE_sum_gt = 0.
        AEE_sum_sum_gt = 0.
        percent_1_AEE_sum = 0.
        percent_3_AEE_sum = 0.
        iters = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(model, batch)
                # pdb.set_trace()

                f_est, f_flow_mask = self.get_estimation_and_target(batch)
                f_gt = f_flow_mask[0]
                # f_mask = f_flow_mask[1]
                event1, _= self.get_input_events(batch)

                # pdb.set_trace()

                AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=self.is_car)
                # AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=True)

                AEE_sum = AEE_sum +  AEE
                AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

                AEE_sum_gt = AEE_sum_gt + AEE_gt
                AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

                percent_1_AEE_sum += percent_1_AEE
                percent_3_AEE_sum += percent_3_AEE

                iters += 1
                # batch_idx1 = int(batch['idx'].cpu().data.numpy()[0]) - 1
                istr = '{:05d} / {:05d}  AEE: {:2.6f}  meanAEE:{:2.6f} 1 - mean %AEE: {:.6f}'.format(batch_idx + 1, len(self.data_loader), AEE, AEE_sum / iters, 1.-percent_1_AEE_sum / iters)

                # print(istr)
                self.logger.write_line(istr, True)
                if(self.save_excel):
                    self.excel.append([batch_idx+1, AEE])

                if iters % self.print_iter == 0:
                    if self.visualize_map:
                        # idx = int(batch['idx'].cpu().data.numpy()[0])
                        idx = batch_idx + 1

                        flow_est = f_est[0].numpy().transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        # self.visualize_optical_flow_light(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt[0].numpy().transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map1, map2 = self.get_key_map(batch)
                        self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                        self.vis_map_RGB(map2, str(idx)+'_map2.jpg')

            if(self.save_excel):
                df = pd.DataFrame(np.array(self.excel), columns=['index','epe'])
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                df.to_csv(os.path.join(self.save_path, "test_epe_per_sample.csv"))

            self.logger.write_line("-------------------------------------------------------", True)
            self.logger.write_line("-----------------Test after {:d} epoch-----------------".format(epoch), True)
            self.logger.write_line("Mean AEE: {:.6f}, sum AEE: {:.6f}, Mean AEE_gt: {:.6f}, sum AEE_gt: {:.6f}, 1 - mean %AEE: {:.6f}, 3 - mean %AEE: {:.6f}, # pts: {:.6f}"
                        .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters,  1.-percent_1_AEE_sum / iters, 1.-percent_3_AEE_sum / iters, n_points), True)
            self.logger.write_line("-------------------------------------------------------", True)

            return AEE_sum / iters
    
    def test_vis_low_epe(self, model, epoch=0, compute_density=False, epe_th=0.5, sequence_list=['indoor_flying2', 'outdoor_day1']):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.eval()
        try:
            model.module.freeze_bn()
        except:
            pass
        
        save_path = self.save_path

        with torch.no_grad():
            for sequence in sequence_list:

                AEE_sum = 0.
                AEE_sum_sum = 0.
                AEE_sum_gt = 0.
                AEE_sum_sum_gt = 0.
                percent_1_AEE_sum = 0.
                percent_3_AEE_sum = 0.
                iters = 0

                self.data_loader.dataset.change_test_sequence(sequence)

                self.save_path = save_path +'/'+ sequence

                excel_sequence = []

                for batch_idx, batch in enumerate(self.data_loader):
                    # Move Data to GPU
                    if next(model.parameters()).is_cuda:
                        batch = self.move_batch_to_cuda(batch)
                    # Network Forward Pass
                    self.run_network(model, batch)

                    f_est, f_flow_mask = self.get_estimation_and_target(batch)
                    f_gt = f_flow_mask[0]
                    # f_mask = f_flow_mask[1]
                    event1, _= self.get_input_events(batch)

                    AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=self.is_car)
                    # AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=True)

                    AEE_sum = AEE_sum +  AEE
                    AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

                    AEE_sum_gt = AEE_sum_gt + AEE_gt
                    AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

                    percent_1_AEE_sum += percent_1_AEE
                    percent_3_AEE_sum += percent_3_AEE

                    iters += 1
                    batch_idx1 = int(batch['idx'].cpu().data.numpy()[0]) - 1
                    istr = '{:05d} / {:05d}  AEE: {:2.6f}  meanAEE:{:2.6f} 1 - mean %AEE: {:.6f}'.format(batch_idx1 + 1, len(self.data_loader), AEE, AEE_sum / iters, 1.-percent_1_AEE_sum / iters)

                    self.logger.write_line(istr, True)
                    if(self.save_excel):
                        if(compute_density):
                            events1, _= self.get_events(batch)
                            event1_den = self.compute_map_density(events1)
                            self.excel.append([sequence+"_"+str(batch_idx1+1), "{:.5f}".format(AEE), "{:.5f}".format(event1_den)])
                            excel_sequence.append([sequence+"_"+str(batch_idx1+1), "{:.5f}".format(AEE), "{:.5f}".format(event1_den)])
                        else:
                            self.excel.append([sequence+"_"+str(batch_idx1+1), "{:.5f}".format(AEE)])
                            excel_sequence.append([sequence+"_"+str(batch_idx1+1), "{:.5f}".format(AEE)])

                    if iters % self.print_iter == 0:
                        if self.visualize_map:
                            
                            if ( AEE == 0 ) or (AEE.detach().numpy() < epe_th):
                                idx = int(batch['idx'].cpu().data.numpy()[0])
                                
                                flow_est = f_est[0].numpy().transpose(1,2,0)
                                self.visualize_optical_flow(flow_est,'{:d}_{:.3f}_flow_est.jpg'.format(idx, AEE))
                                
                                flow_gt = f_gt[0].numpy().transpose(1,2,0)
                                self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                                map1, map2 = self.get_key_map(batch)
                                self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                                self.vis_map_RGB(map2, str(idx)+'_map2.jpg')
                    
                if(self.save_excel):
                    if(compute_density):
                        df = pd.DataFrame(np.array(self.excel), columns=['index','epe', 'density'])
                    else:
                        df = pd.DataFrame(np.array(self.excel), columns=['index','epe'])

                    os.makedirs(save_path, exist_ok=True)
                    df.to_csv(os.path.join(save_path, "sequence_{:s}_test_epe_per_sample.csv".format(sequence)))

            self.save_path = save_path

            if(self.save_excel):
                if(compute_density):
                    df = pd.DataFrame(np.array(self.excel), columns=['index','epe', 'density'])
                else:
                    df = pd.DataFrame(np.array(self.excel), columns=['index','epe'])

                os.makedirs(self.save_path, exist_ok=True)
                df.to_csv(os.path.join(self.save_path, "test_epe_per_sample.csv"))

            self.logger.write_line("-------------------------------------------------------", True)
            self.logger.write_line("-----------------Test after {:d} epoch-----------------".format(epoch), True)
            self.logger.write_line("Mean AEE: {:.6f}, sum AEE: {:.6f}, Mean AEE_gt: {:.6f}, sum AEE_gt: {:.6f}, 1 - mean %AEE: {:.6f}, 3 - mean %AEE: {:.6f}, # pts: {:.6f}"
                        .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters,  1.-percent_1_AEE_sum / iters, 1.-percent_3_AEE_sum / iters, n_points), True)
            self.logger.write_line("-------------------------------------------------------", True)

            return AEE_sum / iters

    def test_multi_sequence(self, model, epoch=0, sequence_list=['indoor_flying2', 'outdoor_day1'], stride=10, vis_events=False, print_epe=False, visualize_map=False):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.eval()
        try:
            model.module.freeze_bn()
        except:
            pass
        
        stride = stride
        self.logger.write_line("test in stride {:d}".format(stride), True)

        save_path = self.save_path
        with torch.no_grad():
            meanAEE = 0.
            meanOut = 0.
            meanAEE_list = []
            meanOut_list = []
            for sequence in sequence_list:
                AEE_sum = 0.
                AEE_sum_sum = 0.
                AEE_sum_gt = 0.
                AEE_sum_sum_gt = 0.
                percent_1_AEE_sum = 0.
                percent_3_AEE_sum = 0.
                iters = 0
                
                self.data_loader.dataset.change_test_sequence(sequence)
                if("outdoor" in sequence):
                    self.is_car = True
                    self.logger.write_line("!!!is_car == True!!!", True)
                else:
                    self.logger.write_line("!!!is_car == False!!!", True)
                    self.is_car = False

                self.print_iter = len(self.data_loader.dataset) / 10

                self.save_path = save_path +'/'+ sequence
                
                for batch_idx, batch in enumerate(self.data_loader):
                    if(batch_idx % stride):
                        continue

                    # Move Data to GPU
                    if next(model.parameters()).is_cuda:
                        batch = self.move_batch_to_cuda(batch)
                    # Network Forward Pass
                    self.run_network(model, batch)
                    # pdb.set_trace()

                    f_est, f_flow_mask = self.get_estimation_and_target(batch)
                    f_gt = f_flow_mask[0]
                    # f_mask = f_flow_mask[1]
                    event_valid = batch["event_valid"].cpu()

                    # pdb.set_trace()

                    AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event_valid, dim=0), dim=0), is_car=self.is_car)
                    # AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=True)

                    AEE_sum = AEE_sum +  AEE
                    AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

                    AEE_sum_gt = AEE_sum_gt + AEE_gt
                    AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

                    percent_1_AEE_sum += percent_1_AEE
                    percent_3_AEE_sum += percent_3_AEE

                    iters += 1
                    # batch_idx1 = int(batch['idx'].cpu().data.numpy()[0]) - 1
                    istr = '{:05d} / {:05d}  AEE: {:2.6f}  meanAEE:{:2.6f} 1 - mean %AEE: {:.6f}'.format(batch_idx + 1, len(self.data_loader), AEE, AEE_sum / iters, 1.-percent_1_AEE_sum / iters)

                    print(istr)
                    # self.logger.write_line(istr, False)

                    if visualize_map:
                        # idx = int(batch['idx'].cpu().data.numpy()[0])
                        idx = batch_idx + 1
                        if(batch_idx % 1 == 0):
                            flow_est = f_est[0].numpy().transpose(1,2,0)
                            if(print_epe):
                                self.visualize_optical_flow(flow_est, str(idx)+'_flow_est_{:.3f}.jpg'.format(AEE))
                            else:
                                # self.visualize_optical_flow_light(flow_est, str(idx)+'_flow_est.jpg')
                                self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')  

                            flow_gt = f_gt[0].numpy().transpose(1,2,0)
                            self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                            map1, map2 = self.get_key_map(batch)
                            if vis_events:
                                events1, events2 = self.get_events(batch)
                                self.vis_map_RGB(events1, str(idx)+'_events1.jpg')
                                self.vis_map_RGB(events2, str(idx)+'_events2.jpg')
                                self.vis_map_RGB(map1, str(idx)+'_emap1.jpg')
                                self.vis_map_RGB(map2, str(idx)+'_emap2.jpg')
                            else:
                                self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                                self.vis_map_RGB(map2, str(idx)+'_map2.jpg')
                self.save_path = save_path
                
                self.logger.write_line("-------------------test_sequence_{:s}------------------".format(sequence), True)
                self.logger.write_line("Mean AEE: {:.6f}, sum AEE: {:.6f}, Mean AEE_gt: {:.6f}, sum AEE_gt: {:.6f}, 1 - mean %AEE: {:.6f}, 3 - mean %AEE: {:.6f}, # pts: {:.6f}"
                        .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters,  1.-percent_1_AEE_sum / iters, 1.-percent_3_AEE_sum / iters, n_points), True)

                meanAEE += AEE_sum / iters
                meanOut += 1.-percent_3_AEE_sum / iters
                meanAEE_list.append(AEE_sum / iters)
                meanOut_list.append(1.-percent_3_AEE_sum / iters)

            self.logger.write_line("-------------------------------------------------------", True)
            self.logger.write_line("-----------------Test after {:d} epoch-----------------".format(epoch), True)
            for i in range(len(sequence_list)):
                self.logger.write_line("{:s}: Mean AEE: {:.6f},  3 - mean %AEE: {:.6f}".format(sequence_list[i], meanAEE_list[i], meanOut_list[i]), True)
            self.logger.write_line("-------------------------------------------------------", True)

            self.logger.write_line("Average points: Mean AEE: {:.6f},  3 - mean %AEE: {:.6f}".format(meanAEE / len(sequence_list), meanOut / len(sequence_list)), True)
            return meanAEE / len(sequence_list)
    
    def test_multi_key(self, model, epoch=0, sequence_list=[],stride=10, vis_events=False, print_epe=False, visualize_map=False):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.eval()
        try:
            model.module.freeze_bn()
        except:
            pass
        
        stride = stride
        self.logger.write_line("test in stride {:d}".format(stride), True)

        save_path = self.save_path

        density_per_index = save_path + '/{:s}.xlsx'.format(self.data_loader.dataset.__class__.__name__)
        writer1 = pd.ExcelWriter(density_per_index)

        total_excel = []

        with torch.no_grad():
            meanAEE = 0.
            meanOut = 0.
            meanAEE_list = []
            meanOut_list = []
            for sequence in sequence_list:
                sequence_excel = []

                AEE_sum = 0.
                AEE_sum_sum = 0.
                AEE_sum_gt = 0.
                AEE_sum_sum_gt = 0.
                percent_1_AEE_sum = 0.
                percent_3_AEE_sum = 0.
                iters = 0
                
                self.data_loader.dataset.change_test_sequence(sequence)
                if("outdoor" in sequence):
                    self.is_car = True
                    self.logger.write_line("!!!is_car == True!!!", True)
                else:
                    self.logger.write_line("!!!is_car == False!!!", True)
                    self.is_car = False

                self.print_iter = len(self.data_loader.dataset) / 10

                self.save_path = save_path +'/'+ sequence
                
                # pdb.set_trace()
                
                for batch_idx, batch in enumerate(self.data_loader):
                    if(batch_idx % stride):
                        continue

                    # Move Data to GPU
                    if next(model.parameters()).is_cuda:
                        batch = self.move_batch_to_cuda(batch)
                    # Network Forward Pass
                    self.run_network(model, batch)
                    # pdb.set_trace()

                    f_est, f_flow_mask = self.get_estimation_and_target(batch)
                    f_gt = f_flow_mask[0]
                    # f_mask = f_flow_mask[1]
                    event_valid = batch['event_valid'].cpu()

                    # pdb.set_trace()

                    AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event_valid, dim=0), dim=0), is_car=self.is_car)
                    # AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=True)

                    AEE_sum = AEE_sum +  AEE
                    AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

                    AEE_sum_gt = AEE_sum_gt + AEE_gt
                    AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

                    percent_1_AEE_sum += percent_1_AEE
                    percent_3_AEE_sum += percent_3_AEE

                    iters += 1
                    # batch_idx1 = int(batch['idx'].cpu().data.numpy()[0]) - 1
                    istr = '{:05d} / {:05d}  AEE: {:2.6f}  meanAEE:{:2.6f} 1 - mean %AEE: {:.6f}'.format(batch_idx + 1, len(self.data_loader), AEE, AEE_sum / iters, 1.-percent_1_AEE_sum / iters)

                    if visualize_map:
                        # idx = int(batch['idx'].cpu().data.numpy()[0])
                        idx = batch_idx + 1

                        flow_est = f_est[0].numpy().transpose(1,2,0)
                        if(print_epe):
                            self.visualize_optical_flow(flow_est, str(idx)+'_flow_est_{:.3f}.jpg'.format(AEE))
                        else:
                            # self.visualize_optical_flow_light(flow_est, str(idx)+'_flow_est.jpg')
                            self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')  

                        flow_gt = f_gt[0].numpy().transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map1, map2 = self.get_key_map(batch)
                        if vis_events:
                            events1, events2 = self.get_events(batch)
                            self.vis_map_RGB(events1, str(idx)+'_events1.jpg')
                            self.vis_map_RGB(events2, str(idx)+'_events2.jpg')
                            self.vis_map_RGB(map1, str(idx)+'_emap1.jpg')
                            self.vis_map_RGB(map2, str(idx)+'_emap2.jpg')
                        else:
                            self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                            self.vis_map_RGB(map2, str(idx)+'_map2.jpg')

                    
                    map1, _ = self.get_key_map(batch)
                    events1, _ = self.get_events(batch)

                    print(map1.abs().min())
                    map1_den = self.compute_map_density(map1)
                    events1_den = self.compute_map_density(events1)
                    # pdb.set_trace()
                    if('d_event_volume_old' in batch.keys()):
                        d_events1, _ = self.get_dense_events(batch)
                        d_events1_den = self.compute_map_density(d_events1)
                        sequence_excel.append(["{:s}_{:d}".format(sequence, batch_idx+1), events1_den, d_events1_den, map1_den, "{:.3f}".format(AEE)])
                        print(["{:s}_{:d}".format(sequence, batch_idx+1),events1_den, d_events1_den, map1_den, "{:.3f}".format(AEE)])
                    else:
                        sequence_excel.append(["{:s}_{:d}".format(sequence, batch_idx+1), events1_den, map1_den, "{:.3f}".format(AEE)])
                        print(["{:s}_{:d}".format(sequence, batch_idx+1), events1_den, map1_den, "{:.3f}".format(AEE)])
                if('d_event_volume_old' in batch.keys()):
                    sequence_df=pd.DataFrame(np.array(sequence_excel), columns=['name','input','dense_event','unet_out','AEE'])
                else:
                    sequence_df=pd.DataFrame(np.array(sequence_excel), columns=['name','input','unet_out','AEE'])
                total_excel.append(sequence_df)    

                self.save_path = save_path
                
                self.logger.write_line("-------------------test_sequence_{:s}------------------".format(sequence), True)
                self.logger.write_line("Mean AEE: {:.6f}, sum AEE: {:.6f}, Mean AEE_gt: {:.6f}, sum AEE_gt: {:.6f}, 1 - mean %AEE: {:.6f}, 3 - mean %AEE: {:.6f}, # pts: {:.6f}"
                        .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters,  1.-percent_1_AEE_sum / iters, 1.-percent_3_AEE_sum / iters, n_points), True)

                meanAEE += AEE_sum / iters
                meanOut += 1.-percent_3_AEE_sum / iters
                meanAEE_list.append(AEE_sum / iters)
                meanOut_list.append(1.-percent_3_AEE_sum / iters)

            pd.concat(total_excel).to_excel(writer1)
            writer1.save()
            writer1.close()

            self.logger.write_line("-------------------------------------------------------", True)
            self.logger.write_line("-----------------Test after {:d} epoch-----------------".format(epoch), True)
            for i in range(len(sequence_list)):
                self.logger.write_line("{:s}: Mean AEE: {:.6f},  3 - mean %AEE: {:.6f}".format(sequence_list[i], meanAEE_list[i], meanOut_list[i]), True)
            self.logger.write_line("-------------------------------------------------------", True)

            self.logger.write_line("Average points: Mean AEE: {:.6f},  3 - mean %AEE: {:.6f}".format(meanAEE / len(sequence_list), meanOut / len(sequence_list)), True)
            return meanAEE / len(sequence_list)

    def test_multi_save_flow(self, model, epoch=0, sequence_list=[],stride=1, vis_events=False, print_epe=False, visualize_map=False):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.eval()
        try:
            model.module.freeze_bn()
        except:
            pass
        
        stride = stride
        self.logger.write_line("test in stride {:d}".format(stride), True)

        save_path = self.save_path

        density_per_index = save_path + '/{:s}.xlsx'.format(self.data_loader.dataset.__class__.__name__)
        writer1 = pd.ExcelWriter(density_per_index)

        total_excel = []

        from utils_luo.tools import file_tools
        
        nori_list_path = save_path + '/flow_est_nori.pkl'
        new_nori_list = {sequence: {} for sequence in sequence_list}

        oss_nori_event = 's3://xinglong/mvsec_ori/MVSEC_all_sequences.nori'
        nsaver = file_tools.Nori_tools.Nori_saver(out_nori_path=oss_nori_event, if_remote=True)
        nsaver.start()


        with torch.no_grad():
            meanAEE = 0.
            meanOut = 0.
            meanAEE_list = []
            meanOut_list = []
            for sequence in sequence_list:
                sequence_excel = []

                AEE_sum = 0.
                AEE_sum_sum = 0.
                AEE_sum_gt = 0.
                AEE_sum_sum_gt = 0.
                percent_1_AEE_sum = 0.
                percent_3_AEE_sum = 0.
                iters = 0
                
                self.data_loader.dataset.change_test_sequence(sequence)
                if("outdoor" in sequence):
                    self.is_car = True
                    self.logger.write_line("!!!is_car == True!!!", True)
                else:
                    self.logger.write_line("!!!is_car == False!!!", True)
                    self.is_car = False

                self.print_iter = len(self.data_loader.dataset) / 10

                self.save_path = save_path +'/'+ sequence
                
                # pdb.set_trace()
                
                for batch_idx, batch in enumerate(self.data_loader):
                    if(batch_idx % stride):
                        continue

                    # Move Data to GPU
                    if next(model.parameters()).is_cuda:
                        batch = self.move_batch_to_cuda(batch)
                    # Network Forward Pass
                    self.run_network(model, batch)
                    # pdb.set_trace()

                    f_est, f_flow_mask = self.get_estimation_and_target(batch)
                    f_gt = f_flow_mask[0]
                    # f_mask = f_flow_mask[1]
                    event1, _= self.get_input_events(batch)

                    # pdb.set_trace()

                    AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=self.is_car)
                    # AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=True)

                    AEE_sum = AEE_sum +  AEE
                    AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

                    AEE_sum_gt = AEE_sum_gt + AEE_gt
                    AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

                    percent_1_AEE_sum += percent_1_AEE
                    percent_3_AEE_sum += percent_3_AEE

                    iters += 1
                    # batch_idx1 = int(batch['idx'].cpu().data.numpy()[0]) - 1
                    istr = '{:05d} / {:05d}  AEE: {:2.6f}  meanAEE:{:2.6f} 1 - mean %AEE: {:.6f}'.format(batch_idx + 1, len(self.data_loader), AEE, AEE_sum / iters, 1.-percent_1_AEE_sum / iters)

                    if visualize_map:
                        # idx = int(batch['idx'].cpu().data.numpy()[0])
                        idx = batch_idx + 1
                        if(idx % 100 == 0):
                            flow_est = f_est[0].numpy().transpose(1,2,0)
                            if(print_epe):
                                self.visualize_optical_flow(flow_est, str(idx)+'_flow_est_{:.3f}.jpg'.format(AEE))
                            else:
                                # self.visualize_optical_flow_light(flow_est, str(idx)+'_flow_est.jpg')
                                self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')  

                            flow_gt = f_gt[0].numpy().transpose(1,2,0)
                            self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                            map1, map2 = self.get_key_map(batch)
                            if vis_events:
                                events1, events2 = self.get_events(batch)
                                self.vis_map_RGB(events1, str(idx)+'_events1.jpg')
                                self.vis_map_RGB(events2, str(idx)+'_events2.jpg')
                                self.vis_map_RGB(map1, str(idx)+'_emap1.jpg')
                                self.vis_map_RGB(map2, str(idx)+'_emap2.jpg')
                            else:
                                self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                                self.vis_map_RGB(map2, str(idx)+'_map2.jpg')

                    name = str(batch['idx'][0])
                    f_est = f_est[0].numpy()
                    flow_id = nsaver.put_file(img=f_est, name=name)
                    new_key = {'flow': flow_id}
                    new_nori_list[sequence].update({name:new_key})
                    
                    map1, _ = self.get_key_map(batch)
                    events1, _ = self.get_events(batch)

                    print(map1.abs().min())
                    map1_den = self.compute_map_density(map1)
                    events1_den = self.compute_map_density(event1)
                    # pdb.set_trace()
                    if('d_event_volume_old' in batch.keys()):
                        d_events1, _ = self.get_dense_events(batch)
                        d_events1_den = self.compute_map_density(d_events1)
                        sequence_excel.append([name, events1_den, d_events1_den, map1_den, "{:.3f}".format(AEE)])
                        print([name,events1_den, d_events1_den, map1_den, "{:.3f}".format(AEE)])
                    else:
                        sequence_excel.append([name, events1_den, map1_den, "{:.3f}".format(AEE)])
                        print([name, events1_den, map1_den, "{:.3f}".format(AEE)])
                if('d_event_volume_old' in batch.keys()):
                    sequence_df=pd.DataFrame(np.array(sequence_excel), columns=['name','input','dense_event','unet_out','AEE'])
                else:
                    sequence_df=pd.DataFrame(np.array(sequence_excel), columns=['name','input','unet_out','AEE'])
                total_excel.append(sequence_df)    

                self.save_path = save_path
                
                self.logger.write_line("-------------------test_sequence_{:s}------------------".format(sequence), True)
                self.logger.write_line("Mean AEE: {:.6f}, sum AEE: {:.6f}, Mean AEE_gt: {:.6f}, sum AEE_gt: {:.6f}, 1 - mean %AEE: {:.6f}, 3 - mean %AEE: {:.6f}, # pts: {:.6f}"
                        .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters,  1.-percent_1_AEE_sum / iters, 1.-percent_3_AEE_sum / iters, n_points), True)

                meanAEE += AEE_sum / iters
                meanOut += 1.-percent_3_AEE_sum / iters
                meanAEE_list.append(AEE_sum / iters)
                meanOut_list.append(1.-percent_3_AEE_sum / iters)

            pd.concat(total_excel).to_excel(writer1)
            writer1.save()
            writer1.close()

            file_tools.pickle_saver.save_pickle(new_nori_list, nori_list_path)
            nsaver.end()
            nsaver.nori_speedup()

            self.logger.write_line("-------------------------------------------------------", True)
            self.logger.write_line("-----------------Test after {:d} epoch-----------------".format(epoch), True)
            for i in range(len(sequence_list)):
                self.logger.write_line("{:s}: Mean AEE: {:.6f},  3 - mean %AEE: {:.6f}".format(sequence_list[i], meanAEE_list[i], meanOut_list[i]), True)
            self.logger.write_line("-------------------------------------------------------", True)

            self.logger.write_line("Average points: Mean AEE: {:.6f},  3 - mean %AEE: {:.6f}".format(meanAEE / len(sequence_list), meanOut / len(sequence_list)), True)
            return meanAEE / len(sequence_list)
    
    def test_multi_density(self, model, epoch, density_list, stride=10, vis_events=False, print_epe=False, visualize_map=True, volumn=False, print_ite=100, model_name="", all_save_path=""):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.eval()
        try:
            model.module.freeze_bn()
        except:
            pass
        
        stride = stride
        self.logger.write_line("test in stride {:d}".format(stride), True)

        if(all_save_path == ""):
            save_path = self.save_path
        else:
            save_path = all_save_path

        with torch.no_grad():
            meanAEE = 0.
            meanOut = 0.
            meanAEE_list = []
            meanOut_list = []
            for sequence in density_list:
                AEE_sum = 0.
                AEE_sum_sum = 0.
                AEE_sum_gt = 0.
                AEE_sum_sum_gt = 0.
                percent_1_AEE_sum = 0.
                percent_3_AEE_sum = 0.
                iters = 0
                
                self.data_loader.dataset.change_test_sequence(sequence)

                self.print_iter = len(self.data_loader.dataset) / 10

                self.save_path = save_path +'/'+ sequence
                
                # pdb.set_trace()
                
                for batch_idx, batch in enumerate(self.data_loader):
                    if(batch_idx % stride):
                        continue

                    # Move Data to GPU
                    if next(model.parameters()).is_cuda:
                        batch = self.move_batch_to_cuda(batch)
                    # Network Forward Pass
                    self.run_network(model, batch)
                    # pdb.set_trace()

                    f_est, f_flow_mask = self.get_estimation_and_target(batch)
                    f_gt = f_flow_mask[0]
                    # f_mask = f_flow_mask[1]

                    names = str(batch['idx'][0])

                    if(volumn):
                        event_volumn = batch['event_volume']
                        event_mask = (torch.sum(torch.sum(torch.sum(event_volumn, dim=0), dim=0), dim=2)).cpu().numpy()[:,:,None]
                        AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, (torch.sum(torch.sum(torch.sum(event_volumn, dim=0), dim=0), dim=2)).cpu(), is_car=self.is_car)

                    else:
                        event1, _= self.get_input_events(batch)
                        event_mask = (torch.sum(torch.sum(event1, dim=0), dim=0)>0).cpu().numpy()[:,:,None]
                        AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=("outdoor" in names))
                        

                    # AEE, percent_1_AEE, percent_3_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = self.flow_error(f_gt, f_est, torch.sum(torch.sum(event1, dim=0), dim=0), is_car=True)

                    AEE_sum = AEE_sum +  AEE
                    AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

                    AEE_sum_gt = AEE_sum_gt + AEE_gt
                    AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

                    percent_1_AEE_sum += percent_1_AEE
                    percent_3_AEE_sum += percent_3_AEE

                    iters += 1
                    # batch_idx1 = int(batch['idx'].cpu().data.numpy()[0]) - 1
                    istr = '{:05d} / {:05d}  AEE: {:2.6f}  meanAEE:{:2.6f} 1 - mean %AEE: {:.6f}'.format(batch_idx + 1, len(self.data_loader), AEE, AEE_sum / iters, 1.-percent_1_AEE_sum / iters)

                    print(istr)
                    # self.logger.write_line(istr, False)

                    if visualize_map:
                        if(batch_idx % print_ite == 0):
                            # idx = int(batch['idx'].cpu().data.numpy()[0])
                            idx = batch_idx + 1

                            flow_est = f_est[0].numpy().transpose(1,2,0)
                            if(print_epe):

                                self.visualize_optical_flow(flow_est, str(idx)+model_name+'_flow_est_{:.3f}.jpg'.format(AEE), events_mask=event_mask)
                                # event_mask = (torch.sum(torch.sum(event1, dim=0), dim=0).numpy())>0
                                # self.visualize_optical_flow(flow_est*event_mask[:,:,None], str(idx)+'_flow_est_masked_{:.3f}.jpg'.format(AEE))
                            else:
                                # self.visualize_optical_flow_light(flow_est, str(idx)+'_flow_est.jpg')
                                self.visualize_optical_flow(flow_est, str(idx)+model_name+'_flow_est.jpg', events_mask=event_mask)
                                # event_mask = (torch.sum(torch.sum(event1, dim=0), dim=0).numpy())>0
                                # self.visualize_optical_flow(flow_est**event_mask[:,:,None], str(idx)+'_flow_est_masked_{:.3f}.jpg'.format(AEE))  

                            flow_gt = f_gt[0].numpy().transpose(1,2,0)
                            self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg', events_mask=event_mask)
                            # event_mask = (torch.sum(torch.sum(event1, dim=0), dim=0).numpy())>0
                            # self.visualize_optical_flow(flow_gt**event_mask[:,:,None], str(idx)+'_flow_gt_masked.jpg'.format(AEE))  

                            map1, map2 = self.get_key_map(batch)
                            if vis_events:
                                events1, events2 = self.get_events(batch)
                                self.vis_map_RGB(events1, str(idx)+'_events1.jpg')
                                self.vis_map_RGB(events2, str(idx)+'_events2.jpg')
                                self.vis_map_RGB(map1, str(idx)+'_emap1.jpg')
                                self.vis_map_RGB(map2, str(idx)+'_emap2.jpg')
                            else:
                                self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                                self.vis_map_RGB(map2, str(idx)+'_map2.jpg')
                                
                self.save_path = save_path
                
                self.logger.write_line("-------------------test_sequence_{:s}------------------".format(sequence), True)
                self.logger.write_line("Mean AEE: {:.6f}, sum AEE: {:.6f}, Mean AEE_gt: {:.6f}, sum AEE_gt: {:.6f}, 1 - mean %AEE: {:.6f}, 3 - mean %AEE: {:.6f}, # pts: {:.6f}"
                        .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters,  1.-percent_1_AEE_sum / iters, 1.-percent_3_AEE_sum / iters, n_points), True)

                meanAEE += AEE_sum / iters
                meanOut += 1.-percent_3_AEE_sum / iters
                meanAEE_list.append(AEE_sum / iters)
                meanOut_list.append(1.-percent_3_AEE_sum / iters)

            self.logger.write_line("-------------------------------------------------------", True)
            self.logger.write_line("-----------------Test after {:d} epoch-----------------".format(epoch), True)
            for i in range(len(density_list)):
                self.logger.write_line("{:s}: Mean AEE: {:.6f},  3 - mean %AEE: {:.6f}".format(density_list[i], meanAEE_list[i], meanOut_list[i]), True)
            self.logger.write_line("-------------------------------------------------------", True)

            self.logger.write_line("Average points: Mean AEE: {:.6f},  3 - mean %AEE: {:.6f}".format(meanAEE / len(density_list), meanOut / len(density_list)), True)
            return meanAEE / len(density_list)

    def inference(self, model):
        model.change_imagesize(self.image_size)
        model.module.freeze_bn()
        model.eval()
        iters = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(batch)
                # pdb.set_trace()

                flow_est, _ = self.get_estimation_and_target(batch)
                flow_est = flow_est[0].numpy().transpose(1,2,0)
                np.save(os.path.join(self.flow_save_path, '{:d}.npy'.format(batch_idx+21)), flow_est)
                
                iters += 1
                istr = '{:05d} / {:05d} '.format(batch_idx + 1, len(self.data_loader))

                self.logger.write_line(istr, True)
    
    def validate_chairs(self, model, epoch=0, iters=24):
        """ Perform evaluation on the FlyingChairs (test) split """
        model.change_imagesize(self.image_size)
        model.cuda()
        model.eval()
        ite = 0
        epe_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(model, batch, iters=iters)
                # pdb.set_trace()

                f_est, f_flow_mask = self.get_estimation_and_target(batch)
                f_gt = f_flow_mask[0]
                # f_mask = f_flow_mask[1]
                gt_flow = f_gt[0]
                pred_flow = f_est[0]

                epe = torch.sum((pred_flow - gt_flow) ** 2, dim=0).sqrt()
                epe_list.append(epe.view(-1).numpy())

                ite += 1
                if ite % 50 == 0:
                    if self.visualize_map:
                        idx = int(batch['idx'].cpu().data.numpy()[0])

                        flow_est = pred_flow.numpy().transpose(1,2,0)
                        self.visualize_optical_flow_light(flow_est, str(idx)+'_flow_est.jpg')

                        map1, map2 = self.get_key_map(batch)
                        self.vis_map(map1, str(idx)+'_map1.jpg')
                        # self.vis_map(map2, str(idx)+'_map2.jpg')
                        # print_str = self.check_tensor(map1, 'unet_out1')
                        # self.logger.write_line(print_str, True)
                        
                        mask1, mask2 = self.get_key_mask(batch)
                        if mask1 is not None:
                            self.vis_mask(mask1, str(idx)+'_mask1.jpg')
                            # self.vis_mask(mask2, str(idx)+'_mask2.jpg')
                            key_map1 = map1 * mask1
                            self.vis_map(key_map1, str(idx)+'_key_map1.jpg')

            epe = np.mean(np.concatenate(epe_list))
            self.logger.write_line("-------------------------------------------------------", True)
            self.logger.write_line("-----------------Test after {:d} epoch-----------------".format(epoch), True)
            self.logger.write_line("Validation Chairs EPE: {:f}".format(epe), True)
            self.logger.write_line("-------------------------------------------------------", True)
        
        return epe

            
class TestRaftEvents(Test):

    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'valid' in batch.keys():
                return batch['flow_est'].cpu().data, (batch['flow'].cpu().data, batch['valid'].cpu().data)
            return batch['flow_est'].cpu().data, batch['flow'].cpu().data
        else:
            f_est = batch['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch['flow'].cpu().data, scale_factor=0.5)
            if 'valid' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['valid'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def get_input_events(self, batch):
        if not self.downsample:
            im1 = batch['event_volume_old'].cpu().data
            im2 = batch['event_volume_new'].cpu().data
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'].cpu().data, scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'].cpu().data, scale_factor=0.5)
        return im1, im2   

    def get_key_map(self, batch):
        if(isinstance(batch['map_list'][-1], dict)):
            map1 = {}
            for key, value in batch['map_list'][0].items():
                map1.update({key: value.cpu().data})
            
            map2 = {}
            for key, value in batch['map_list'][1].items():
                map2.update({key: value.cpu().data})

        elif(isinstance(batch['map_list'][-1], list)):
            map1 = batch['map_list'][-1][0].cpu().data
            map2 = batch['map_list'][-1][1].cpu().data

        else:
            map1 = batch['map_list'][0].cpu().data
            map2 = batch['map_list'][1].cpu().data

        return map1, map2 
    
    def get_key_mask(self, batch):
        if batch['map_list'][2] is not None:
            if not self.downsample:
                mask1 = batch['map_list'][2].cpu().data
                mask2 = batch['map_list'][3].cpu().data
            else:
                mask1 = torch.nn.functional.interpolate(batch['map_list'][2].cpu().data, scale_factor=0.5)
                mask2 = torch.nn.functional.interpolate(batch['map_list'][3].cpu().data, scale_factor=0.5)
            return mask1, mask2
        else:
            return None, None    

    def run_network(self, model, batch, iters=12):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        # batch['map_list'], batch['flow_list'] = model(im1, im2, iters=iters, normal=self.normal)
        batch['map_list'], batch['flow_list'] = model(events1=im1, events2=im2)
        
        batch['flow_est'] = batch['flow_list'][-1]


class TestDenseSparse(Test):

    def get_input_events(self, batch):
        if not self.downsample:
            im1 = batch['event_volume_old'].cpu().data
            im2 = batch['event_volume_new'].cpu().data
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'].cpu().data, scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'].cpu().data, scale_factor=0.5)
        return im1, im2   

    def get_key_map(self, batch):
        if not self.downsample:
            if(isinstance(batch['map_list'][-1], list)):
                map1 = batch['map_list'][-1][0].cpu().data
                map2 = batch['map_list'][-1][1].cpu().data
            else:
                map1 = batch['map_list'][0].cpu().data
                map2 = batch['map_list'][1].cpu().data
        else:
            map1 = torch.nn.functional.interpolate(batch['map_list'][0].cpu().data, scale_factor=0.5)
            map2 = torch.nn.functional.interpolate(batch['map_list'][1].cpu().data, scale_factor=0.5)
        return map1, map2 

    def get_key_mask(self, batch):
        if batch['map_list'][2] is not None:
            if not self.downsample:
                mask1 = batch['map_list'][2].cpu().data
                mask2 = batch['map_list'][3].cpu().data
            else:
                mask1 = torch.nn.functional.interpolate(batch['map_list'][2].cpu().data, scale_factor=0.5)
                mask2 = torch.nn.functional.interpolate(batch['map_list'][3].cpu().data, scale_factor=0.5)
            return mask1, mask2
        else:
            return None, None    

    def run_network(self, model, batch, iters=12):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        # batch['map_list'], batch['flow_list'] = model(im1, im2, iters=iters, normal=self.normal)
        batch['map_list'], batch['flow_list'] = model(events1=im1, events2=im2, iters=iters, normal=self.normal)
        
        batch['flow_est'] = batch['flow_list'][-1]
    

