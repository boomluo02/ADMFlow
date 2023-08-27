import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

class ImagePadder(object):
    # =================================================================== #
    # In some networks, the image gets downsized. This is a problem, if   #
    # the to-be-downsized image has odd dimensions ([15x20]->[7.5x10]).   #
    # To prevent this, the input image of the network needs to be a       #
    # multiple of a minimum size (min_size)                               #
    # The ImagePadder makes sure, that the input image is of such a size, #
    # and if not, it pads the image accordingly.                          #
    # =================================================================== #

    def __init__(self, min_size=64):
        # --------------------------------------------------------------- #
        # The min_size additionally ensures, that the smallest image      #
        # does not get too small                                          #
        # --------------------------------------------------------------- #
        self.min_size = min_size
        self.pad_height = None
        self.pad_width = None

    def pad(self, image):
        # --------------------------------------------------------------- #
        # If necessary, this function pads the image on the left & top    #
        # --------------------------------------------------------------- #
        height, width = image.shape[-2:]
        if self.pad_width is None:
            self.pad_height = (self.min_size - height % self.min_size)%self.min_size
            self.pad_width = (self.min_size - width % self.min_size)%self.min_size
        else:
            pad_height = (self.min_size - height % self.min_size)%self.min_size
            pad_width = (self.min_size - width % self.min_size)%self.min_size
            if pad_height != self.pad_height or pad_width != self.pad_width:
                raise
        return nn.ZeroPad2d((self.pad_width, 0, self.pad_height, 0))(image)

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height:, self.pad_width:]

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
