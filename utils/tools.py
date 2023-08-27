import torch.optim as optim
import imageio
import cv2
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn.init import xavier_normal, kaiming_normal
from torch.utils.data import Dataset
import pickle
import argparse
import collections
import random
from shutil import rmtree
import time
import zipfile
import png
import array
import warnings
import shutil
from PIL import Image
import re
from collections import Iterable
from matplotlib.colors import hsv_to_rgb


if torch.__version__ in ['1.1.0', ]:
    from torch.utils.data.dataloader import _DataLoaderIter, DataLoader
elif torch.__version__ in ['1.2.0', '1.4.0', '1.5.1', '1.6.0', '1.7.0','1.10.0','1.10.1','1.12.0']:
    from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as _DataLoaderIter
    from torch.utils.data.dataloader import DataLoader
else:
    raise ValueError('torch version error: %s' % torch.__version__)


class tools():
    class abs_database():
        def __init__(self):
            self.data_ls = {'train': [], 'val': [], 'test': []}
            self.len = {'train': 0, 'val': 0, 'test': 0}
            self.len_train = 0
            self.len_val = 0
            self.len_test = 0

        def sample(self, index, split):
            pass

        def _init_len(self):
            self.len = {'train': len(self.data_ls['train']), 'val': len(self.data_ls['val']), 'test': len(self.data_ls['test'])}
            self.len_train = self.len['train']
            self.len_val = self.len['train']
            self.len_test = self.len['train']

    class abstract_config():
        @classmethod
        def _check_length_of_file_name(cls, file_name):
            if len(file_name) >= 255:
                return False
            else:
                return True

        @classmethod
        def _check_length_of_file_path(cls, filepath):
            if len(filepath) >= 4096:
                return False
            else:
                return True

        @property
        def to_dict(self):
            def dict_class(obj):
                temp = {}
                k = dir(obj)
                for name in k:
                    if not name.startswith('_') and name != 'to_dict':
                        value = getattr(obj, name)
                        if callable(value):
                            pass
                        else:
                            temp[name] = value
                return temp

            s_dict = dict_class(self)
            return s_dict

        @property
        def _key_list(self):
            k_list = list(self.to_dict.keys())
            return k_list

        def update(self, data: dict):

            t_key = list(data.keys())
            for i in self._key_list:
                if i in t_key:
                    setattr(self, i, data[i])
                    print('set param ====  %s:   %s' % (i, data[i]))

        def __contains__(self, item):
            '''  use to check something in config '''
            if item in self._key_list:
                return True
            else:
                return False

        def print_defaut_dict(self):
            d = self.to_dict
            l = self._key_list
            l = sorted(l)
            for i in l:
                value = d[i]
                if type(value) == str:
                    temp = "'%s'" % value
                else:
                    temp = value
                print("'%s':%s," % (i, temp))

        @classmethod
        def __demo(cls):
            class temp(tools.abstract_config):
                def __init__(self, **kwargs):
                    self.if_gpu = True
                    self.eval_batch_size = 1
                    self.eval_name = 'flyingchairs'
                    self.eval_datatype = 'nori'  # or base
                    self.if_print_process = False

                    self.update(kwargs)

            a = temp(eval_batch_size=8, eval_name='flyingchairs', eval_datatype='nori', if_print_process=False)

    class abstract_model(nn.Module):

        def save_model(self, save_path):
            torch.save(self.state_dict(), save_path)

        def load_model(self, load_path, if_relax=False, if_print=True):
            if if_print:
                print('loading protrained model from %s' % load_path)
            if if_relax:
                model_dict = self.state_dict()
                pretrained_dict = torch.load(load_path)
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                pretrained_dict_v2 = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            pretrained_dict_v2[k] = v
                model_dict.update(pretrained_dict_v2)
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(torch.load(load_path))

        def load_from_model(self, model: nn.Module, if_relax=False):
            if if_relax:
                model_dict = self.state_dict()
                pretrained_dict = model.state_dict()
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                pretrained_dict_v2 = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            pretrained_dict_v2[k] = v
                model_dict.update(pretrained_dict_v2)
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(model.state_dict())

        def choose_gpu(self, gpu_opt=None):
            # choose gpu
            if gpu_opt is None:
                # gpu=0
                model = self.cuda()
                # torch.cuda.set_device(gpu)
                # model.cuda(gpu)
                # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
                # print('torch.cuda.device_count()  ',torch.cuda.device_count())
                # model=torch.nn.parallel.DistributedDataParallel(model,device_ids=range(torch.cuda.device_count()))
                model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  # multi gpu
            elif gpu_opt == 0:
                model = self.cuda()
            else:
                if type(gpu_opt) != int:
                    raise ValueError('wrong gpu config:  %s' % (str(gpu_opt)))
                torch.cuda.set_device(gpu_opt)
                model = self.cuda(gpu_opt)
            return model

        @classmethod
        def save_model_gpu(cls, model, path):
            name_dataparallel = torch.nn.DataParallel.__name__
            if type(model).__name__ == name_dataparallel:
                model = model.module
            model.save_model(path)

    class abs_test_model(abstract_model):
        def __init__(self):
            super(tools.abs_test_model, self).__init__()
            self.result_save_dir = None
            self.some_save_results = False
            self.some_ids = None  # only save some results
            self.id_cnt = -1
            self.eval_id_scores = {}
            self.id_cnt_save_dir = ''

        def prepare_eval(self):
            self.id_cnt += 1
            if self.result_save_dir is not None:
                save_flag = True
                self.id_cnt_save_dir = os.path.join(self.result_save_dir, '%s' % self.id_cnt)
                if self.some_save_results and self.id_cnt not in self.some_ids:
                    eval_flag = False
                else:
                    file_tools.check_dir(self.id_cnt_save_dir)
                    eval_flag = True
            else:
                save_flag = False
                eval_flag = True
            return eval_flag, save_flag

        def eval_forward(self, im1, im2, *args, **kwargs):  # do model forward and cache forward results
            self.id_cnt += 1
            if self.some_ids is None:
                pass
            else:
                pass
            return 0

        def do_save_results(self, result_save_dir=None, some_save_results=False):
            self.result_save_dir = result_save_dir
            if result_save_dir is not None:
                file_tools.check_dir(result_save_dir)
            self.some_save_results = some_save_results
            # define some id
            if result_save_dir is not None:
                self.some_ids = [7 * 7 * i + 1 for i in range(24)]  # [1,50,99,148, ..., 981, 1030, 1079, 1128]

        def record_eval_score(self, eval_score):
            self.eval_id_scores[self.id_cnt] = eval_score
            if os.path.isdir(self.id_cnt_save_dir):
                new_dir_name = '%s_EPE_%.2f' % (self.id_cnt, float(eval_score))
                os.renames(self.id_cnt_save_dir, os.path.join(self.result_save_dir, new_dir_name))

        def save_record(self):
            print('eval results saved at: %s' % self.result_save_dir)
            file_tools.pickle_saver.save_pickle(self.eval_id_scores, os.path.join(self.result_save_dir, 'scores.pkl'))
            self.id_cnt = -1

    class data_prefetcher():

        def __init__(self, dataset, gpu_opt=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, batch_gpu_index=0):
            self.dataset = dataset
            loader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory)
            # self.loader = iter(loader)
            self.loader = _DataLoaderIter(loader)
            self.stream = torch.cuda.Stream()
            self.gpu_opt = gpu_opt

            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.drop_last = drop_last
            self.batch_gpu_index = batch_gpu_index

        def build(self):
            loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, drop_last=self.drop_last, pin_memory=self.pin_memory)
            self.loader = _DataLoaderIter(loader)
            # self.loader = iter(loader)

        def next(self):
            try:
                # batch = next(self.loader)
                batch = self.loader.next()
            except StopIteration:
                self.build()
                return None
            # print('self.batch',type(self.batch))
            # for i in range(len(self.batch)):
            #     print('i',i,type(self.batch[i]))
            with torch.cuda.stream(self.stream):
                cpu_batch, gpu_batch = batch[:self.batch_gpu_index], batch[self.batch_gpu_index:]
                gpu_batch = tensor_tools.tensor_gpu(*gpu_batch, check_on=True, non_blocking=True, gpu_opt=self.gpu_opt)
                batch = cpu_batch + gpu_batch
                # self.next_img = self.next_img.cuda(non_blocking=True).float()
                # self.next_seg = self.next_seg.cuda(non_blocking=True).float()
                # self.next_weight = self.next_weight.cuda(non_blocking=True)
                # self.mask2 = self.mask2.cuda(non_blocking=True).float()
                # self.mask3 = self.mask3.cuda(non_blocking=True).float()

                # With Amp, it isn't necessary to manually convert data to half.
                # if args.fp16:
                #     self.next_input = self.next_input.half()
                # else:
                # self.next_input = self.next_input.float()
                # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            return batch

    class data_prefetcher_dict():

        def __init__(self, dataset, gpu_keys, gpu_opt=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
            self.dataset = dataset
            loader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory)
            self.gpu_keys = gpu_keys  # keys in batches to be loaded to gpu, e.g. gpu_keys=('im1', 'im2')
            # self.loader = iter(loader)
            self.loader = _DataLoaderIter(loader)
            self.stream = torch.cuda.Stream()
            self.gpu_opt = gpu_opt

            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.drop_last = drop_last

        def build(self):
            loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, drop_last=self.drop_last, pin_memory=self.pin_memory)
            self.loader = _DataLoaderIter(loader)
            # self.loader = iter(loader)

        def next(self):
            try:
                # batch = next(self.loader)
                batch = self.loader.next()
            except StopIteration:
                self.build()
                return None
            with torch.cuda.stream(self.stream):
                for i in self.gpu_keys:
                    batch[i] = self.check_on_gpu(batch[i], non_blocking=True)
            return batch

        def check_on_gpu(self, tensor_, non_blocking=True):
            if type(self.gpu_opt) == int:
                tensor_g = tensor_.cuda(self.gpu_opt, non_blocking=non_blocking)
            else:
                tensor_g = tensor_.cuda()
            return tensor_g

    class DataProvider:

        def __init__(self, dataset, batch_size, shuffle=True, num_worker=4, drop_last=True, pin_memory=True):
            self.batch_size = batch_size
            self.dataset = dataset
            self.dataiter = None
            self.iteration = 0  #
            self.epoch = 0  #
            self.shuffle = shuffle
            self.pin_memory = pin_memory
            self.num_worker = num_worker
            self.drop_last = drop_last

        def build(self):
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_worker,
                                    pin_memory=self.pin_memory,
                                    drop_last=self.drop_last)
            self.dataiter = _DataLoaderIter(dataloader)

        def next(self):
            if self.dataiter is None:
                self.build()
            try:
                batch = self.dataiter.next()
                self.iteration += 1

                # if self.is_cuda:
                #     batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
                return batch

            except StopIteration:  # ??epoch???reload
                self.epoch += 1
                self.build()
                self.iteration = 1  # reset and return the 1st batch

                batch = self.dataiter.next()
                # if self.is_cuda:
                #     batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
                return batch

    class AverageMeter():

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, num):
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

    class Avg_meter_ls():
        def __init__(self):
            self.data_ls = {}
            self.short_name_ls = {}

        def update(self, name, val, num, short_name=None):
            if name not in self.data_ls.keys():
                self.data_ls[name] = tools.AverageMeter()
                if short_name is None:
                    short_name = name
                self.short_name_ls[name] = short_name
            self.data_ls[name].update(val=val, num=num)

        def print_loss(self, name):
            a = ' %s %.4f(%.4f)' % (self.short_name_ls[name], self.data_ls[name].val, self.data_ls[name].avg)
            return a

        def print_avg_loss(self, name):
            a = ' %s: %.4f' % (self.short_name_ls[name], self.data_ls[name].avg)
            return a

        def print_all_losses(self):
            a = ''
            for i in sorted(self.data_ls.keys()):
                a += ' %s %.4f(%.4f)' % (self.short_name_ls[i], self.data_ls[i].val, self.data_ls[i].avg)
            return a

        def print_all_losses_final(self):
            a = ''
            for i in sorted(self.data_ls.keys()):
                a += ' %s=%.4f' % (self.short_name_ls[i], self.data_ls[i].avg)
            return a

        def get_all_losses_final(self):
            a = {}
            for i in sorted(self.data_ls.keys()):
                a[i] = self.data_ls[i].avg
            return a

        def reset(self):
            for name in self.data_ls.keys():
                self.data_ls[name].reset()

    class TimeClock():

        def __init__(self):
            self.st = 0
            self.en = 0
            self.start_flag = False

        def start(self):
            self.reset()
            self.start_flag = True
            self.st = time.time()

        def reset(self):
            self.start_flag = False
            self.st = 0
            self.en = 0

        def end(self):
            self.en = time.time()

        def get_during(self):
            return self.en - self.st

    # 研究一下图像加字体展示结果
    class Text_img():
        def __init__(self, **kwargs):
            self.font = 'simplex'
            self.my_font_type = 'black_white'
            self.__update(kwargs)
            self.font_ls = {
                'simplex': cv2.FONT_HERSHEY_SIMPLEX,
                'plain': cv2.FONT_HERSHEY_PLAIN,
                'complex': cv2.FONT_HERSHEY_COMPLEX,
                'trplex': cv2.FONT_HERSHEY_TRIPLEX,
                # 'complex_small': cv2.FONT_HERSHEY_COMPLEX_SMALL,
                'italic': cv2.FONT_ITALIC,
            }
            self.my_font_type_ls = {
                'black_white': self._black_white,
            }
            self.show_func = self.my_font_type_ls[self.my_font_type]

        def __update(self, data: dict):
            def dict_class(obj):
                temp = {}
                k = dir(obj)
                for name in k:
                    if not name.startswith('_'):
                        value = getattr(obj, name)
                        if callable(value):
                            pass
                        else:
                            temp[name] = value
                return temp

            s_dict = dict_class(self)
            k_list = list(s_dict.keys())
            t_key = list(data.keys())
            for i in k_list:
                if i in t_key:
                    setattr(self, i, data[i])
                    # print('set param ====  %s:   %s' % (i, data[i]))

        def _black_white(self, img, text, scale, row=0):
            # params
            color_1 = (10, 10, 10)
            thick_1 = 5
            color_2 = (255, 255, 255)
            thick_2 = 2

            # get position: Bottom-left
            t_w, t_h, t_inter = self._check_text_size(text=text, scale=scale, thick=thick_1)
            pw = t_inter
            ph = t_h + t_inter + row * (t_h + t_inter)

            # put text
            img_ = img.copy()
            img_ = cv2.putText(img_, text, (pw, ph), fontFace=self.font_ls[self.font], fontScale=scale, color=color_1, thickness=thick_1)
            img_ = cv2.putText(img_, text, (pw, ph), fontFace=self.font_ls[self.font], fontScale=scale, color=color_2, thickness=thick_2)
            return img_

        def _check_text_size(self, text: str, scale=1, thick=1):
            textSize, baseline = cv2.getTextSize(text, self.font_ls[self.font], scale, thick)
            twidth, theight = textSize
            return twidth, theight, baseline // 2

        def put_text(self, img, text=None, scale=1):
            if text is not None:
                if type(text) == str:
                    img = self.show_func(img, text, scale, 0)
                elif isinstance(text, Iterable):
                    for i, t in enumerate(text):
                        img = self.show_func(img, t, scale, i)
            return img

        def draw_cross(self, img, point_wh, cross_length=5, color=(0, 0, 255)):  #
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1]), (point_wh[0] + cross_length, point_wh[1]), color, thick)
            new_img = cv2.line(new_img, (point_wh[0], point_wh[1] - cross_length), (point_wh[0], point_wh[1] + cross_length), color, thick)
            return new_img

        def draw_cross_black_white(self, img, point_wh, cross_length=5):  #
            if cross_length <= 5:
                cross_length = 5
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1]), (point_wh[0] + cross_length, point_wh[1]), (0, 0, 0), thick)
            new_img = cv2.line(new_img, (point_wh[0], point_wh[1] - cross_length), (point_wh[0], point_wh[1] + cross_length), (0, 0, 0), thick)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1]), (point_wh[0] + cross_length, point_wh[1]), (250, 250, 250), thick // 2)
            new_img = cv2.line(new_img, (point_wh[0], point_wh[1] - cross_length), (point_wh[0], point_wh[1] + cross_length), (250, 250, 250), thick // 2)
            return new_img

        def draw_x(self, img, point_wh, cross_length=5, color=(0, 0, 255)):
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] - cross_length), (point_wh[0] + cross_length, point_wh[1] + cross_length), color, thick)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] + cross_length), (point_wh[0] + cross_length, point_wh[1] - cross_length), color, thick)
            return new_img

        def draw_x_black_white(self, img, point_wh, cross_length=5):
            if cross_length <= 5:
                cross_length = 5
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] - cross_length), (point_wh[0] + cross_length, point_wh[1] + cross_length), (0, 0, 0), thick)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] + cross_length), (point_wh[0] + cross_length, point_wh[1] - cross_length), (0, 0, 0), thick)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] - cross_length), (point_wh[0] + cross_length, point_wh[1] + cross_length), (250, 250, 250), thick // 2)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] + cross_length), (point_wh[0] + cross_length, point_wh[1] - cross_length), (250, 250, 250), thick // 2)
            return new_img

        def demo(self):
            im = np.ones((500, 500, 3), dtype='uint8') * 50
            imshow = self.put_text(im, text=list('demo show sample text'.split(' ')), scale=1)
            cv2.imshow('im', imshow)
            cv2.waitKey()


class file_tools():

    class flow_read_write():

        @classmethod
        def write_flow_png(cls, filename, uv, v=None, mask=None):

            if v is None:
                assert (uv.ndim == 3)
                assert (uv.shape[2] == 2)
                u = uv[:, :, 0]
                v = uv[:, :, 1]
            else:
                u = uv

            assert (u.shape == v.shape)

            height_img, width_img = u.shape
            if mask is None:
                valid_mask = np.ones([height_img, width_img], dtype=np.uint16)
            else:
                valid_mask = mask

            flow_u = np.clip((u * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)
            flow_v = np.clip((v * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)

            output = np.stack((flow_u, flow_v, valid_mask), axis=-1)

            with open(filename, 'wb') as f:
                # writer = png.Writer(width=width_img, height=height_img, bitdepth=16)
                # temp = np.reshape(output, (-1, width_img * 3))
                # writer.write(f, temp)

                png_writer = png.Writer(width=width_img, height=height_img, bitdepth=16, compression=3, greyscale=False)
                # png_writer.write_array(f, output)
                temp = np.reshape(output, (-1, width_img * 3))
                png_writer.write(f, temp)

        @classmethod
        def write_kitti_png_file(cls, flow_fn, flow_data, mask_data=None):
            flow_img = np.zeros((flow_data.shape[0], flow_data.shape[1], 3),
                                dtype=np.uint16)
            if mask_data is None:
                mask_data = np.ones([flow_data.shape[0], flow_data.shape[1]], dtype=np.uint16)
            flow_img[:, :, 2] = (flow_data[:, :, 0] * 64.0 + 2 ** 15).astype(np.uint16)
            flow_img[:, :, 1] = (flow_data[:, :, 1] * 64.0 + 2 ** 15).astype(np.uint16)
            flow_img[:, :, 0] = mask_data[:, :]
            cv2.imwrite(flow_fn, flow_img)

        @classmethod
        def read_flo(cls, filename):
            with open(filename, 'rb') as f:
                magic = np.fromfile(f, np.float32, count=1)
                if 202021.25 != magic:
                    print('Magic number incorrect. Invalid .flo file')
                else:
                    w = np.fromfile(f, np.int32, count=1)
                    h = np.fromfile(f, np.int32, count=1)
                    data = np.fromfile(f, np.float32, count=int(2 * w * h))
                    # Reshape data into 3D array (columns, rows, bands)
                    data2D = np.resize(data, (h[0], w[0], 2))
                    return data2D

        @classmethod
        def write_flo(cls, flow, filename):
            """
            write optical flow in Middlebury .flo format
            :param flow: optical flow map
            :param filename: optical flow file path to be saved
            :return: None
            """
            f = open(filename, 'wb')
            magic = np.array([202021.25], dtype=np.float32)
            (height, width) = flow.shape[0:2]
            w = np.array([width], dtype=np.int32)
            h = np.array([height], dtype=np.int32)
            magic.tofile(f)
            w.tofile(f)
            h.tofile(f)
            flow.tofile(f)
            f.close()

        @classmethod
        def point_vec(cls, img, flow, valid=None):
            meshgrid = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
            rate_x = 1
            rate_y = 1
            dispimg = cv2.resize(img, None, fx=rate_x, fy=rate_y)
            # colorflow = tools.flow_to_image(flow).astype(int)
            colorflow = tensor_tools.flow_to_image_ndmax(flow)
            if valid is None:
                valid = np.ones((img.shape[0], img.shape[1]), dtype=flow.dtype)
            for i in range(img.shape[1]):  # x
                for j in range(img.shape[0]):  # y
                    # if flow[j, i, 2] != 1: continue
                    if valid[j, i] != 1: continue
                    if j % 20 != 0 or i % 20 != 0: continue
                    xend = int((meshgrid[0][j, i] + flow[j, i, 0]) * rate_x)
                    yend = int((meshgrid[1][j, i] + flow[j, i, 1]) * rate_y)
                    leng = np.linalg.norm(flow[j, i, :2])
                    if leng < 1: continue
                    dispimg = cv2.arrowedLine(dispimg, (meshgrid[0][j, i] * rate_y, meshgrid[1][j, i] * rate_x), (xend, yend),
                                            (int(colorflow[j, i, 0]), int(colorflow[j, i, 1]), int(colorflow[j, i, 2])), 1,
                                            tipLength=8 / leng, line_type=cv2.LINE_AA)
            return dispimg

    @classmethod
    def check_dir(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def tryremove(cls, name, file=False):
        try:
            if file:
                os.remove(name)
            else:
                rmtree(name)
        except OSError:
            pass

    @classmethod
    def extract_zip(cls, zip_path, extract_dir):
        print('unzip file: %s' % zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)


class tensor_tools():

    class nianjin_warp():

        @classmethod
        def get_grid(cls, batch_size, H, W, start):
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            ones = torch.ones_like(xx)
            grid = torch.cat((xx, yy, ones), 1).float()
            if torch.cuda.is_available():
                grid = grid.cuda()
            # print("grid",grid.shape)
            # print("start", start)
            grid[:, :2, :, :] = grid[:, :2, :, :] + start  # 加上patch在原图内的偏移量

            return grid

        @classmethod
        def transformer(cls, I, vgrid, train=True):
            # I: Img, shape: batch_size, 1, full_h, full_w
            # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
            # outsize: (patch_h, patch_w)

            def _repeat(x, n_repeats):

                rep = torch.ones([n_repeats, ]).unsqueeze(0)
                rep = rep.int()
                x = x.int()

                x = torch.matmul(x.reshape([-1, 1]), rep)
                return x.reshape([-1])

            def _interpolate(im, x, y, out_size, scale_h):
                # x: x_grid_flat
                # y: y_grid_flat
                # out_size: same as im.size
                # scale_h: True if normalized
                # constants
                num_batch, num_channels, height, width = im.size()

                out_height, out_width = out_size[0], out_size[1]
                # zero = torch.zeros_like([],dtype='int32')
                zero = 0
                max_y = height - 1
                max_x = width - 1
                if scale_h:
                    # scale indices from [-1, 1] to [0, width or height]
                    # print('--Inter- scale_h:', scale_h)
                    x = (x + 1.0) * (height) / 2.0
                    y = (y + 1.0) * (width) / 2.0

                # do sampling
                x0 = torch.floor(x).int()
                x1 = x0 + 1
                y0 = torch.floor(y).int()
                y1 = y0 + 1

                x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
                x1 = torch.clamp(x1, zero, max_x)
                y0 = torch.clamp(y0, zero, max_y)
                y1 = torch.clamp(y1, zero, max_y)

                dim1 = torch.from_numpy(np.array(width * height))
                dim2 = torch.from_numpy(np.array(width))

                base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)  # 其实就是单纯标出batch中每个图的下标位置
                # base = torch.arange(0,num_batch) * dim1
                # base = base.reshape(-1, 1).repeat(1, out_height * out_width).reshape(-1).int()
                # 区别？expand不对数据进行拷贝 .reshape(-1,1).expand(-1,out_height * out_width).reshape(-1)
                if torch.cuda.is_available():
                    dim2 = dim2.cuda()
                    dim1 = dim1.cuda()
                    y0 = y0.cuda()
                    y1 = y1.cuda()
                    x0 = x0.cuda()
                    x1 = x1.cuda()
                    base = base.cuda()
                base_y0 = base + y0 * dim2
                base_y1 = base + y1 * dim2
                idx_a = base_y0 + x0
                idx_b = base_y1 + x0
                idx_c = base_y0 + x1
                idx_d = base_y1 + x1

                # use indices to lookup pixels in the flat image and restore
                # channels dim
                im = im.permute(0, 2, 3, 1)
                im_flat = im.reshape([-1, num_channels]).float()

                idx_a = idx_a.unsqueeze(-1).long()
                idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
                Ia = torch.gather(im_flat, 0, idx_a)

                idx_b = idx_b.unsqueeze(-1).long()
                idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
                Ib = torch.gather(im_flat, 0, idx_b)

                idx_c = idx_c.unsqueeze(-1).long()
                idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
                Ic = torch.gather(im_flat, 0, idx_c)

                idx_d = idx_d.unsqueeze(-1).long()
                idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
                Id = torch.gather(im_flat, 0, idx_d)

                # and finally calculate interpolated values
                x0_f = x0.float()
                x1_f = x1.float()
                y0_f = y0.float()
                y1_f = y1.float()

                wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
                wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
                wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
                wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
                output = wa * Ia + wb * Ib + wc * Ic + wd * Id

                return output

            def _transform(I, vgrid, scale_h):

                C_img = I.shape[1]
                B, C, H, W = vgrid.size()

                x_s_flat = vgrid[:, 0, ...].reshape([-1])
                y_s_flat = vgrid[:, 1, ...].reshape([-1])
                out_size = vgrid.shape[2:]
                input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

                output = input_transformed.reshape([B, H, W, C_img])
                return output

            # scale_h = True
            output = _transform(I, vgrid, scale_h=False)
            if train:
                output = output.permute(0, 3, 1, 2)
            return output

        @classmethod
        def warp_im(cls, I_nchw, flow_nchw, start_n211):
            batch_size, _, img_h, img_w = I_nchw.size()
            _, _, patch_size_h, patch_size_w = flow_nchw.size()
            patch_indices = cls.get_grid(batch_size, patch_size_h, patch_size_w, start_n211)
            vgrid = patch_indices[:, :2, ...]
            # grid_warp = vgrid - flow_nchw
            grid_warp = vgrid + flow_nchw
            pred_I2 = cls.transformer(I_nchw, grid_warp)
            return pred_I2

    class occ_check_model():

        def __init__(self, occ_type='for_back_check', occ_alpha_1=1.0, occ_alpha_2=0.05, obj_out_all='all'):
            self.occ_type_ls = ['for_back_check', 'forward_warp', 'for_back_check&forward_warp']
            assert occ_type in self.occ_type_ls
            assert obj_out_all in ['obj', 'out', 'all']
            self.occ_type = occ_type
            self.occ_alpha_1 = occ_alpha_1
            self.occ_alpha_2 = occ_alpha_2
            self.sum_abs_or_squar = False
            self.obj_out_all = obj_out_all

        def __call__(self, flow_f, flow_b, scale=1):
            # 输入进来是可使用的光流

            if self.obj_out_all == 'all':
                if self.occ_type == 'for_back_check':
                    occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, scale=scale)
                elif self.occ_type == 'forward_warp':
                    raise ValueError('not implemented')
                elif self.occ_type == 'for_back_check&forward_warp':
                    raise ValueError('not implemented')
                else:
                    raise ValueError('occ type should be in %s, get %s' % (self.occ_type_ls, self.occ_type))
                return occ_1, occ_2
            elif self.obj_out_all == 'obj':
                if self.occ_type == 'for_back_check':
                    occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, scale=scale)
                elif self.occ_type == 'forward_warp':
                    raise ValueError('not implemented')
                elif self.occ_type == 'for_back_check&forward_warp':
                    raise ValueError('not implemented')
                else:
                    raise ValueError('occ type should be in %s, get %s' % (self.occ_type_ls, self.occ_type))
                out_occ_fw = self.torch_outgoing_occ_check(flow_f)
                out_occ_bw = self.torch_outgoing_occ_check(flow_b)
                obj_occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_1, out_occ=out_occ_fw)
                obj_occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_2, out_occ=out_occ_bw)
                return obj_occ_fw, obj_occ_bw
            elif self.obj_out_all == 'out':
                out_occ_fw = self.torch_outgoing_occ_check(flow_f)
                out_occ_bw = self.torch_outgoing_occ_check(flow_b)
                return out_occ_fw, out_occ_bw
            else:
                raise ValueError("obj_out_all should be in ['obj','out','all'], but get: %s" % self.obj_out_all)

        def _forward_backward_occ_check(self, flow_fw, flow_bw, scale=1):
            """
            In this function, the parameter alpha needs to be improved
            """

            def length_sq_v0(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.pow(temp, 0.5)
                return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                # return temp

            def length_sq(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.pow(temp, 0.5)
                # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                return temp

            if self.sum_abs_or_squar:
                sum_func = length_sq_v0
            else:
                sum_func = length_sq
            mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
            flow_bw_warped = tensor_tools.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
            flow_fw_warped = tensor_tools.torch_warp(flow_fw, flow_bw)
            flow_diff_fw = flow_fw + flow_bw_warped
            flow_diff_bw = flow_bw + flow_fw_warped
            occ_thresh = self.occ_alpha_1 * mag_sq + self.occ_alpha_2 / scale
            occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
            occ_bw = sum_func(flow_diff_bw) < occ_thresh
            # if IF_DEBUG:
            #     temp_ = sum_func(flow_diff_fw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
            #     temp_ = sum_func(flow_diff_bw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
            #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
            #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
            return occ_fw.float(), occ_bw.float()

        def forward_backward_occ_check(self, flow_fw, flow_bw, alpha1, alpha2, obj_out_all='obj'):
            """
            In this function, the parameter alpha needs to be improved
            """

            def length_sq_v0(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.pow(temp, 0.5)
                return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                # return temp

            def length_sq(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.pow(temp, 0.5)
                # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                return temp

            if self.sum_abs_or_squar:
                sum_func = length_sq_v0
            else:
                sum_func = length_sq
            mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
            flow_bw_warped = tensor_tools.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
            flow_fw_warped = tensor_tools.torch_warp(flow_fw, flow_bw)
            flow_diff_fw = flow_fw + flow_bw_warped
            flow_diff_bw = flow_bw + flow_fw_warped
            occ_thresh = alpha1 * mag_sq + alpha2
            occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
            occ_bw = sum_func(flow_diff_bw) < occ_thresh
            occ_fw = occ_fw.float()
            occ_bw = occ_bw.float()
            # if IF_DEBUG:
            #     temp_ = sum_func(flow_diff_fw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
            #     temp_ = sum_func(flow_diff_bw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
            #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
            #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
            if obj_out_all == 'obj':
                out_occ_fw = self.torch_outgoing_occ_check(flow_fw)
                out_occ_bw = self.torch_outgoing_occ_check(flow_bw)
                occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_fw, out_occ=out_occ_fw)
                occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_bw, out_occ=out_occ_bw)
            return occ_fw, occ_bw

        def _forward_warp_occ_check(self, flow_bw):  # TODO
            return 0

        @classmethod
        def torch_outgoing_occ_check(cls, flow):

            B, C, H, W = flow.size()
            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
            flow_x, flow_y = torch.split(flow, 1, 1)
            if flow.is_cuda:
                xx = xx.cuda()
                yy = yy.cuda()
            # tools.check_tensor(flow_x, 'flow_x')
            # tools.check_tensor(flow_y, 'flow_y')
            # tools.check_tensor(xx, 'xx')
            # tools.check_tensor(yy, 'yy')
            pos_x = xx + flow_x
            pos_y = yy + flow_y
            # tools.check_tensor(pos_x, 'pos_x')
            # tools.check_tensor(pos_y, 'pos_y')
            # print(' ')
            # check mask
            outgoing_mask = torch.ones_like(pos_x)
            outgoing_mask[pos_x > W - 1] = 0
            outgoing_mask[pos_x < 0] = 0
            outgoing_mask[pos_y > H - 1] = 0
            outgoing_mask[pos_y < 0] = 0
            return outgoing_mask.float()

        @classmethod
        def torch_get_obj_occ_check(cls, occ_mask, out_occ):
            outgoing_mask = torch.zeros_like(occ_mask)
            if occ_mask.is_cuda:
                outgoing_mask = outgoing_mask.cuda()
            outgoing_mask[occ_mask == 1] = 1
            outgoing_mask[out_occ == 0] = 1
            return outgoing_mask

    # Part of the code from https://github.com/visinf/irr/blob/master/augmentations.py  ## Portions of Code from, copyright 2018 Jochen Gast
    class Interpolation():
        @classmethod
        def _bchw2bhwc(cls, tensor):
            return tensor.transpose(1, 2).transpose(2, 3)

        @classmethod
        def _bhwc2bchw(cls, tensor):
            return tensor.transpose(2, 3).transpose(1, 2)

        @classmethod
        def resize2D(cls, inputs, size_targets, mode="bilinear"):
            size_inputs = [inputs.size(2), inputs.size(3)]

            if all([size_inputs == size_targets]):
                return inputs  # nothing to do
            elif any([size_targets < size_inputs]):
                resized = F.adaptive_avg_pool2d(inputs, size_targets)  # downscaling
            else:
                resized = F.upsample(inputs, size=size_targets, mode=mode)  # upsampling

            # correct scaling
            return resized

        @classmethod
        def resize2D_as(cls, inputs, output_as, mode="bilinear"):
            size_targets = [output_as.size(2), output_as.size(3)]
            return tensor_tools.Interpolation.resize2D(inputs, size_targets, mode=mode)

        class Meshgrid(nn.Module):
            def __init__(self):
                super(tensor_tools.Interpolation.Meshgrid, self).__init__()
                self.width = 0
                self.height = 0
                self.register_buffer("xx", torch.zeros(1, 1))
                self.register_buffer("yy", torch.zeros(1, 1))
                self.register_buffer("rangex", torch.zeros(1, 1))
                self.register_buffer("rangey", torch.zeros(1, 1))

            def _compute_meshgrid(self, width, height):
                torch.arange(0, width, out=self.rangex)
                torch.arange(0, height, out=self.rangey)
                self.xx = self.rangex.repeat(height, 1).contiguous()
                self.yy = self.rangey.repeat(width, 1).t().contiguous()

            def forward(self, width, height):
                if self.width != width or self.height != height:
                    self._compute_meshgrid(width=width, height=height)
                    self.width = width
                    self.height = height
                return self.xx, self.yy

        class BatchSub2Ind(nn.Module):
            def __init__(self):
                super(tensor_tools.Interpolation.BatchSub2Ind, self).__init__()
                self.register_buffer("_offsets", torch.LongTensor())

            def forward(self, shape, row_sub, col_sub, out=None):
                batch_size = row_sub.size(0)
                height, width = shape
                ind = row_sub * width + col_sub
                torch.arange(batch_size, out=self._offsets)
                self._offsets *= (height * width)

                if out is None:
                    return torch.add(ind, self._offsets.view(-1, 1, 1))
                else:
                    torch.add(ind, self._offsets.view(-1, 1, 1), out=out)

        class Interp2(nn.Module):
            def __init__(self, clamp=False):
                super(tensor_tools.Interpolation.Interp2, self).__init__()
                self._clamp = clamp
                self._batch_sub2ind = tensor_tools.Interpolation.BatchSub2Ind()
                self.register_buffer("_x0", torch.LongTensor())
                self.register_buffer("_x1", torch.LongTensor())
                self.register_buffer("_y0", torch.LongTensor())
                self.register_buffer("_y1", torch.LongTensor())
                self.register_buffer("_i00", torch.LongTensor())
                self.register_buffer("_i01", torch.LongTensor())
                self.register_buffer("_i10", torch.LongTensor())
                self.register_buffer("_i11", torch.LongTensor())
                self.register_buffer("_v00", torch.FloatTensor())
                self.register_buffer("_v01", torch.FloatTensor())
                self.register_buffer("_v10", torch.FloatTensor())
                self.register_buffer("_v11", torch.FloatTensor())
                self.register_buffer("_x", torch.FloatTensor())
                self.register_buffer("_y", torch.FloatTensor())

            def forward(self, v, xq, yq):
                batch_size, channels, height, width = v.size()

                # clamp if wanted
                if self._clamp:
                    xq.clamp_(0, width - 1)
                    yq.clamp_(0, height - 1)

                # ------------------------------------------------------------------
                # Find neighbors
                #
                # x0 = torch.floor(xq).long(),          x0.clamp_(0, width - 1)
                # x1 = x0 + 1,                          x1.clamp_(0, width - 1)
                # y0 = torch.floor(yq).long(),          y0.clamp_(0, height - 1)
                # y1 = y0 + 1,                          y1.clamp_(0, height - 1)
                #
                # ------------------------------------------------------------------
                self._x0 = torch.floor(xq).long().clamp(0, width - 1)
                self._y0 = torch.floor(yq).long().clamp(0, height - 1)

                self._x1 = torch.add(self._x0, 1).clamp(0, width - 1)
                self._y1 = torch.add(self._y0, 1).clamp(0, height - 1)

                # batch_sub2ind
                self._batch_sub2ind([height, width], self._y0, self._x0, out=self._i00)
                self._batch_sub2ind([height, width], self._y0, self._x1, out=self._i01)
                self._batch_sub2ind([height, width], self._y1, self._x0, out=self._i10)
                self._batch_sub2ind([height, width], self._y1, self._x1, out=self._i11)

                # reshape
                v_flat = tensor_tools.Interpolation._bchw2bhwc(v).contiguous().view(-1, channels)
                torch.index_select(v_flat, dim=0, index=self._i00.view(-1), out=self._v00)
                torch.index_select(v_flat, dim=0, index=self._i01.view(-1), out=self._v01)
                torch.index_select(v_flat, dim=0, index=self._i10.view(-1), out=self._v10)
                torch.index_select(v_flat, dim=0, index=self._i11.view(-1), out=self._v11)

                # local_coords
                torch.add(xq, - self._x0.float(), out=self._x)
                torch.add(yq, - self._y0.float(), out=self._y)

                # weights
                w00 = torch.unsqueeze((1.0 - self._y) * (1.0 - self._x), dim=1)
                w01 = torch.unsqueeze((1.0 - self._y) * self._x, dim=1)
                w10 = torch.unsqueeze(self._y * (1.0 - self._x), dim=1)
                w11 = torch.unsqueeze(self._y * self._x, dim=1)

                def _reshape(u):
                    return tensor_tools.Interpolation._bhwc2bchw(u.view(batch_size, height, width, channels))

                # values
                values = _reshape(self._v00) * w00 + _reshape(self._v01) * w01 \
                         + _reshape(self._v10) * w10 + _reshape(self._v11) * w11

                if self._clamp:
                    return values
                else:
                    #  find_invalid
                    invalid = ((xq < 0) | (xq >= width) | (yq < 0) | (yq >= height)).unsqueeze(dim=1).float()
                    # maskout invalid
                    transformed = invalid * torch.zeros_like(values) + (1.0 - invalid) * values

                return transformed

    @classmethod
    def torch_warp_mask(cls, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        mask = torch.autograd.Variable(torch.ones(x.size()))
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output, mask

    @classmethod
    def torch_warp(cls, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        # mask = torch.autograd.Variable(torch.ones(x.size()))
        # if x.is_cuda:
        #     mask = mask.cuda()
        # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        # output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output

    @classmethod
    def torch_warp_boundary(cls, x, flo, start_point):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        start_point: [B,2,1,1]
        """

        _, _, Hx, Wx = x.size()
        B, C, H, W = flo.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo + start_point

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(Wx - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(Hx - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        # mask = torch.autograd.Variable(torch.ones(x.size()))
        # if x.is_cuda:
        #     mask = mask.cuda()
        # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        # output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output

    @classmethod
    def weights_init(cls, m):
        classname = m.__class__.__name__
        if classname.find('conv') != -1:
            # torch.nn.init.xavier_normal(m.weight)
            torch.nn.init.kaiming_normal(m.weight)

            torch.nn.init.constant(m.bias, 0)

    @classmethod
    def create_gif(cls, image_list, gif_name, duration=0.5):
        frames = []
        for image_name in image_list:
            frames.append(image_name)
        imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
        return

    @classmethod
    def warp_cv2(cls, img_prev, flow):
        # calculate mat
        w = int(img_prev.shape[1])
        h = int(img_prev.shape[0])
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.float32(np.dstack([x_coords, y_coords]))
        pixel_map = coords + flow
        new_frame = cv2.remap(img_prev, pixel_map, None, cv2.INTER_LINEAR)
        return new_frame

    @classmethod
    def flow_to_image_dmax(cls, flow, display=False):
        """

        :param flow: H,W,2
        :param display:
        :return: H,W,3
        """

        def compute_color(u, v):
            def make_color_wheel():
                """
                Generate color wheel according Middlebury color code
                :return: Color wheel
                """
                RY = 15
                YG = 6
                GC = 4
                CB = 11
                BM = 13
                MR = 6

                ncols = RY + YG + GC + CB + BM + MR

                colorwheel = np.zeros([ncols, 3])

                col = 0

                # RY
                colorwheel[0:RY, 0] = 255
                colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
                col += RY

                # YG
                colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
                colorwheel[col:col + YG, 1] = 255
                col += YG

                # GC
                colorwheel[col:col + GC, 1] = 255
                colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
                col += GC

                # CB
                colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
                colorwheel[col:col + CB, 2] = 255
                col += CB

                # BM
                colorwheel[col:col + BM, 2] = 255
                colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
                col += + BM

                # MR
                colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
                colorwheel[col:col + MR, 0] = 255

                return colorwheel

            """
            compute optical flow color map
            :param u: optical flow horizontal map
            :param v: optical flow vertical map
            :return: optical flow in color code
            """
            [h, w] = u.shape
            img = np.zeros([h, w, 3])
            nanIdx = np.isnan(u) | np.isnan(v)
            u[nanIdx] = 0
            v[nanIdx] = 0

            colorwheel = make_color_wheel()
            ncols = np.size(colorwheel, 0)

            rad = np.sqrt(u ** 2 + v ** 2)

            a = np.arctan2(-v, -u) / np.pi

            fk = (a + 1) / 2 * (ncols - 1) + 1

            k0 = np.floor(fk).astype(int)

            k1 = k0 + 1
            k1[k1 == ncols + 1] = 1
            f = fk - k0

            for i in range(0, np.size(colorwheel, 1)):
                tmp = colorwheel[:, i]
                col0 = tmp[k0 - 1] / 255
                col1 = tmp[k1 - 1] / 255
                col = (1 - f) * col0 + f * col1

                idx = rad <= 1
                col[idx] = 1 - rad[idx] * (1 - col[idx])
                notidx = np.logical_not(idx)

                col[notidx] *= 0.75
                img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

            return img

        UNKNOWN_FLOW_THRESH = 1e7
        """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

        if display:
            print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)

        img = compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)

    @classmethod
    def flow_to_image_ndmax(cls, flow, max_flow=None):
        # flow shape (H, W, C)
        if max_flow is not None:
            max_flow = max(max_flow, 1.)
        else:
            max_flow = np.max(flow)

        n = 8
        u, v = flow[:, :, 0], flow[:, :, 1]
        mag = np.sqrt(np.square(u) + np.square(v))
        angle = np.arctan2(v, u)
        im_h = np.mod(angle / (2 * np.pi) + 1, 1)
        im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
        im_v = np.clip(n - im_s, a_min=0, a_max=1)
        im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
        return (im * 255).astype(np.uint8)

    @classmethod
    def flow_error_image_np(cls, flow_pred, flow_gt, mask_occ, mask_noc=None, log_colors=True):
        """Visualize the error between two flows as 3-channel color image.
        Adapted from the KITTI C++ devkit.
        Args:
            flow_pred: prediction flow of shape [ height, width, 2].
            flow_gt: ground truth
            mask_occ: flow validity mask of shape [num_batch, height, width, 1].
                Equals 1 at (occluded and non-occluded) valid pixels.
            mask_noc: Is 1 only at valid pixels which are not occluded.
        """
        # mask_noc = tf.ones(tf.shape(mask_occ)) if mask_noc is None else mask_noc
        mask_noc = np.ones(mask_occ.shape) if mask_noc is None else mask_noc
        diff_sq = (flow_pred - flow_gt) ** 2
        # diff = tf.sqrt(tf.reduce_sum(diff_sq, [3], keep_dims=True))
        diff = np.sqrt(np.sum(diff_sq, axis=2, keepdims=True))
        if log_colors:
            height, width, _ = flow_pred.shape
            # num_batch, height, width, _ = tf.unstack(tf.shape(flow_1))
            colormap = [
                [0, 0.0625, 49, 54, 149],
                [0.0625, 0.125, 69, 117, 180],
                [0.125, 0.25, 116, 173, 209],
                [0.25, 0.5, 171, 217, 233],
                [0.5, 1, 224, 243, 248],
                [1, 2, 254, 224, 144],
                [2, 4, 253, 174, 97],
                [4, 8, 244, 109, 67],
                [8, 16, 215, 48, 39],
                [16, 1000000000.0, 165, 0, 38]]
            colormap = np.asarray(colormap, dtype=np.float32)
            colormap[:, 2:5] = colormap[:, 2:5] / 255
            # mag = tf.sqrt(tf.reduce_sum(tf.square(flow_2), 3, keep_dims=True))
            tempp = np.square(flow_gt)
            # temp = np.sum(tempp, axis=2, keep_dims=True)
            # mag = np.sqrt(temp)
            mag = np.sqrt(np.sum(tempp, axis=2, keepdims=True))
            # error = tf.minimum(diff / 3, 20 * diff / mag)
            error = np.minimum(diff / 3, 20 * diff / (mag + 1e-7))
            im = np.zeros([height, width, 3])
            for i in range(colormap.shape[0]):
                colors = colormap[i, :]
                cond = np.logical_and(np.greater_equal(error, colors[0]), np.less(error, colors[1]))
                # temp=np.tile(cond, [1, 1, 3])
                im = np.where(np.tile(cond, [1, 1, 3]), np.ones([height, width, 1]) * colors[2:5], im)
            # temp=np.cast(mask_noc, np.bool)
            # im = np.where(np.tile(np.cast(mask_noc, np.bool), [1, 1, 3]), im, im * 0.5)
            im = np.where(np.tile(mask_noc == 1, [1, 1, 3]), im, im * 0.5)
            im = im * mask_occ
        else:
            error = (np.minimum(diff, 5) / 5) * mask_occ
            im_r = error  # errors in occluded areas will be red
            im_g = error * mask_noc
            im_b = error * mask_noc
            im = np.concatenate([im_r, im_g, im_b], axis=2)
            # im = np.concatenate(axis=2, values=[im_r, im_g, im_b])
        return im[:, :, ::-1]

    @classmethod
    def tensor_gpu(cls, *args, check_on=True, gpu_opt=None, non_blocking=True):
        def check_on_gpu(tensor_):
            if type(gpu_opt) == int:
                tensor_g = tensor_.cuda(gpu_opt, non_blocking=non_blocking)
            else:
                tensor_g = tensor_.cuda()
            return tensor_g

        def check_off_gpu(tensor_):
            if tensor_.is_cuda:
                tensor_c = tensor_.cpu()
            else:
                tensor_c = tensor_
            tensor_c = tensor_c.detach().numpy()
            # tensor_c = cv2.normalize(tensor_c.detach().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return tensor_c

        if torch.cuda.is_available():
            if check_on:
                data_ls = [check_on_gpu(a) for a in args]
            else:
                data_ls = [check_off_gpu(a) for a in args]
        else:
            if check_on:
                data_ls = args
            else:
                # data_ls = args
                data_ls = [a.detach().numpy() for a in args]
                # data_ls = [cv2.normalize(a.detach().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) for a in args]
                # data_ls = args
        return data_ls

    @classmethod
    def cv2_show_dict(cls, **kwargs):
        for i in kwargs.keys():
            cv2.imshow(i, kwargs[i])
        cv2.waitKey()

    @classmethod
    def hist_match_np_hw3(cls, img, ref):
        '''need BGR image input'''
        # channels = ['blue', 'green', 'red']
        out = np.zeros_like(img)
        _, _, colorChannel = img.shape
        for i in range(colorChannel):
            # print(channels[i])
            hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
            hist_ref, _ = np.histogram(ref[:, :, i], 256)
            cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
            cdf_ref = np.cumsum(hist_ref)

            for j in range(256):
                tmp = abs(cdf_img[j] - cdf_ref)
                tmp = tmp.tolist()
                idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
                out[:, :, i][img[:, :, i] == j] = idx
        return out
        # cv2.imwrite('0.jpg', out)
        # print('Done')

    @classmethod
    def hist_match_np_3hw(cls, img, ref):
        '''need BGR image input'''
        # channels = ['blue', 'green', 'red']
        out = np.zeros_like(img)
        colorChannel, _, _ = img.shape
        for i in range(colorChannel):
            # print(channels[i])
            hist_img, _ = np.histogram(img[i, :, :], 256)  # get the histogram
            hist_ref, _ = np.histogram(ref[i, :, :], 256)
            cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
            cdf_ref = np.cumsum(hist_ref)

            for j in range(256):
                tmp = abs(cdf_img[j] - cdf_ref)
                tmp = tmp.tolist()
                idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
                out[i, :, :][img[i, :, :] == j] = idx
        return out
        # cv2.imwrite('0.jpg', out)
        # print('Done')

    @classmethod
    def compute_model_size(cls, model, *args):
        from thop import profile
        flops, params = profile(model, inputs=args, verbose=False)
        print('flops: %.3f G, params: %.3f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

    @classmethod
    def count_parameters(cls, model):
        a = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return a

    @classmethod
    def im_norm(cls, img):
        eps = 1e-6
        a = np.max(img)
        b = np.min(img)
        if a - b <= 0:
            img = (img - b) / (a - b + eps)
        else:
            img = (img - b) / (a - b)
        img = img * 255
        img = img.astype('uint8')
        return img

    @classmethod
    def check_tensor(cls, data, name, print_data=False, print_in_txt=None):
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

    @classmethod
    def check_tensor_np(cls, data, name, print_data=False, print_in_txt=None):
        temp = data
        a = len(name)
        name_ = name + ' ' * 100
        name_ = name_[0:max(a, 10)]
        print_str = '%s, %s, %s, %s,%s,%s,%s' % (name_, temp.shape, data.dtype, ' max:%.2f' % np.max(temp), ' min:%.2f' % np.min(temp),
                                                 ' mean:%.2f' % np.mean(temp), ' sum:%.2f' % np.sum(temp))
        if print_in_txt is None:
            print(print_str)
        else:
            print(print_str, file=print_in_txt)
        if print_data:
            print(temp)
        return print_str


class frame_utils():
    '''  borrowed from RAFT '''
    TAG_CHAR = np.array([202021.25], np.float32)

    @classmethod
    def readFlow(cls, fn):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

    @classmethod
    def readPFM(cls, file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

    @classmethod
    def writeFlow(cls, filename, uv, v=None):
        """ Write optical flow to file.

        If v is None, uv is assumed to contain both u and v channels,
        stacked in depth.
        Original code by Deqing Sun, adapted from Daniel Scharstein.
        """
        nBands = 2

        if v is None:
            assert (uv.ndim == 3)
            assert (uv.shape[2] == 2)
            u = uv[:, :, 0]
            v = uv[:, :, 1]
        else:
            u = uv

        assert (u.shape == v.shape)
        height, width = u.shape
        f = open(filename, 'wb')
        # write the header
        f.write(cls.TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        f.close()

    @classmethod
    def readFlowKITTI(cls, filename):
        flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        flow = flow[:, :, ::-1].astype(np.float32)
        flow, valid = flow[:, :, :2], flow[:, :, 2]
        flow = (flow - 2 ** 15) / 64.0
        return flow, valid

    @classmethod
    def read_png_flow(cls, fpath):
        """
        Read KITTI optical flow, returns u,v,valid mask

        """

        R = png.Reader(fpath)
        width, height, data, _ = R.asDirect()
        # This only worked with python2.
        # I = np.array(map(lambda x:x,data)).reshape((height,width,3))
        gt = np.array([x for x in data]).reshape((height, width, 3))
        flow = gt[:, :, 0:2]
        flow = (flow.astype('float64') - 2 ** 15) / 64.0
        flow = flow.astype(np.float)
        mask = gt[:, :, 2:3]
        mask = np.uint8(mask)
        return flow, mask

    @classmethod
    def readDispKITTI(cls, filename):
        disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
        valid = disp > 0.0
        flow = np.stack([-disp, np.zeros_like(disp)], -1)
        return flow, valid

    @classmethod
    def writeFlowKITTI(cls, filename, uv):
        uv = 64.0 * uv + 2 ** 15
        valid = np.ones([uv.shape[0], uv.shape[1], 1])
        uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
        cv2.imwrite(filename, uv[..., ::-1])

    @classmethod
    def read_gen(cls, file_name, read_mask=False):
        ext = os.path.splitext(file_name)[-1]
        if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
            if read_mask:
                return imageio.imread(file_name)
            else:
                return Image.open(file_name)
        elif ext == '.bin' or ext == '.raw':
            return np.load(file_name)
        elif ext == '.flo':
            return cls.readFlow(file_name).astype(np.float32)
        elif ext == '.pfm':
            flow = cls.readPFM(file_name).astype(np.float32)
            if len(flow.shape) == 2:
                return flow
            else:
                return flow[:, :, :-1]
        else:
            raise ValueError('wrong file type: %s' % ext)


if __name__ == '__main__':
    a_np = np.ones((1, 3, 100, 100))
    m = np.ones((1, 1, 100, 100))
    print(a_np.shape)
    a = torch.from_numpy(a_np)
    b = torch.from_numpy(a_np)
    m = torch.from_numpy(m)
    loss_diff, occ_weight = Loss_tools.weighted_ssim(a, b, m)
    ssim = torch.sum(loss_diff * occ_weight) / (torch.sum(occ_weight) + 1e-6)
    print(ssim)
