import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock
from model_utils import coords_grid, upflow8, ImagePadder

from adm import MDC, MDS

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def get_args():
    from argparse import Namespace
    args = Namespace(small=False,
                     dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args

class ADMFlow(nn.Module):
    def __init__(self, config, n_first_channels=5):
        # args:
        super(ADMFlow, self).__init__()
        args = get_args()
        self.args = args
        self.image_padder = ImagePadder(min_size=32)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # unet network
        self.in_channels = n_first_channels

        self.unet = MDC(self.in_channels, self.in_channels)
        self.unet_sk = MDS(self.in_channels)

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=0,
                                    n_first_channels=self.in_channels)
        
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.resnet = True
        if("without_res" in config.keys()):
            self.resnet = False
    
    def change_imagesize(self, img_size):
        self.image_size = img_size
        # self.image_padder = InputPadder(img_size, mode='chairs')

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, events1, events2, iters=12, flow_init=None, upsample=True, normal=False):

        events_p1 = self.image_padder.pad(events1)
        events_p2 = self.image_padder.pad(events2)

        events_p1 = events_p1.contiguous()
        events_p2 = events_p2.contiguous()

        unet_out = self.unet([events_p1, events_p2], with_res=self.resnet)

        image1, image2 = self.unet_sk([events_p1, events_p2], [unet_out[-1][0], unet_out[-1][1]])

    
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):

            cnet = self.cnet(image2)

            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.image_padder.unpad(flow_up))
        
        map_out = []
        for maps in unet_out:
            map1, map2 = maps[0], maps[1]
            map_out.append([self.image_padder.unpad(map1), self.image_padder.unpad(map2)])

        return map_out, flow_predictions
    
    @classmethod
    def demo(cls):
        im = torch.zeros((3, 5, 256, 256))
        config = {}
        net = ADMFlow(config, n_first_channels=5)
        maps,out = net(im, im)
        print(maps[0][0].shape)
        print(out[0].shape)


if __name__ == '__main__':

    ADMFlow.demo()