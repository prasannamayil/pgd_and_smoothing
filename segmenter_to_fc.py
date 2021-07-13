"""
Need to add felzenswalb also to these seg_slics
"""


## model.py Imports
import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np

from utils import * ## For seg_SLIC


##<======================================= Different functions to convert img ==========================================>

def imgs2tensor(imgs, args):
    ''' Takes in  image tensor (Cuda) and spits out fc tensor.Converts segmenter to tensor for a batch of images that
    can be added on top of the standard net (function only works for SC dataset, needs work for standard datasets).
    Not using this anymore.
    '''

    n = args['number_of_segments']
    dim = int(args['image_size']**2)
    nchans = args['number_of_channels']
    device = args['device']

    temp = np.zeros((imgs.shape[0], dim * nchans, dim * nchans))
    for obs in range(imgs.shape[0]):
        segs = seg_SLIC(imgs[obs, :].cpu(), n)

        segs = segs.flatten()
        fc_weight = np.zeros((len(segs) * nchans, len(segs) * nchans))

        if n == dim:
            fc_weight = np.identity(dim * nchans)
        else:
            for i in np.unique(segs):
                locations = np.where(segs == i)[0]
                number_of_pixels = len(locations)
                temp_col = np.tile(locations, number_of_pixels)
                temp_row = np.repeat(locations, number_of_pixels)

                col_idx = list(temp_col)
                row_idx = list(temp_row)
                for c in range(nchans - 1):
                    col_idx.extend(list(temp_col + (dim * (c + 1))))
                    row_idx.extend(list(temp_row + (dim * (c + 1))))

                fc_weight[col_idx, row_idx] = 1 / number_of_pixels

        temp[obs, :] = fc_weight
    temp = torch.Tensor(temp).to(device)
    return temp


def img2fc(img, args):
    ''' Takes in single  image (Cuda) and spits out fc tensor. Converts segmenter for one image to fully connected layer
    Not using this anymore
    '''
    n = args['number_of_segments']
    dim = int(args['image_size']**2)
    nchans = args['number_of_channels']
    device = args['device']

    segs = seg_SLIC(img.cpu(), n)

    segs = segs.flatten()
    fc_weight = np.zeros((len(segs) * nchans, len(segs) * nchans))

    if n == dim:
        fc_weight = np.identity(dim * nchans)
    else:
        for i in np.unique(segs):
            number_of_pixels = len(np.where(segs == i)[0])
            sc = 1 / number_of_pixels
            col_idx = np.tile(np.where(segs == i)[0], number_of_pixels)
            row_idx = np.repeat(np.where(segs == i)[0], number_of_pixels)
            for c in range(nchans):
                fc_weight[(col_idx + dim * c, row_idx + dim * c)] = sc

    out = nn.Linear(dim * nchans, dim * nchans).to(device)
    out.bias.data = torch.zeros_like(out.bias.data)
    out.weight.data = torch.FloatTensor(fc_weight).to(device)
    return out


def imgs2tensor_fast(net, imgs, args):
    ''' Takes in batch of  images and updates net's fcAvg.
    Not using this anymore
    '''
    n = args['number_of_segments']
    dim = int(args['image_size']**2)
    nchans = args['number_of_channels']

    net.fcAvg *= 0
    for obs in range(imgs.shape[0]):
        segs = seg_SLIC(imgs[obs, :].cpu(), n)
        segs = segs.flatten()
        if n == dim:
            net.fcAvg[obs, np.arange(dim * nchans), np.arange(dim * nchans)] = 1
        else:
            for i in np.unique(segs):
                locations = np.where(segs == i)[0]
                number_of_pixels = len(locations)
                temp_col = np.tile(locations, number_of_pixels)
                temp_row = np.repeat(locations, number_of_pixels)

                if nchans == 3:
                    col_idx = np.hstack((temp_col, temp_col + dim, temp_col + dim * 2))
                    row_idx = np.hstack((temp_row, temp_row + dim, temp_row + dim * 2))
                elif nchans == 1:
                    col_idx = temp_col
                    row_idx = temp_row
                else:
                    print("something's wrong")
                    break
                net.fcAvg[obs, col_idx, row_idx] = 1 / number_of_pixels


def imgs2tensor_faster(net, imgs, args):
    ''' Takes in batch of  images and updates net's fcAvg. This is the fastest function.
    Using this atm.
    '''

    n = args['number_of_segments']
    dim = int(args['image_size']**2)
    nchans = args['number_of_channels']

    net.fcAvg *= 0  ## Very important

    obs_segs = {}
    for obs in range(imgs.shape[0]):
        obs_segs[obs] = {}
        segs = seg_SLIC(imgs[obs, :].cpu(), n)
        segs = segs.flatten()
        if n == dim:
            net.fcAvg[obs, np.arange(dim * nchans), np.arange(dim * nchans)] = 1
        else:
            for i in np.unique(segs):
                obs_segs[obs][i] = {}
                locations = np.where(segs == i)[0]
                number_of_pixels = len(locations)
                temp_col = np.tile(locations, number_of_pixels)
                temp_row = np.repeat(locations, number_of_pixels)

                if nchans == 3:
                    col_idx = np.hstack((temp_col, temp_col + dim, temp_col + dim * 2))
                    row_idx = np.hstack((temp_row, temp_row + dim, temp_row + dim * 2))
                elif nchans == 1:
                    col_idx = temp_col
                    row_idx = temp_row
                else:
                    print("something's wrong")
                    break
                obs_segs[obs][i]['col'] = col_idx
                obs_segs[obs][i]['row'] = row_idx
                obs_segs[obs][i]['sc'] = 1 / number_of_pixels

    ## Set values of tensor
    if n != dim:
        for obs in obs_segs.keys():
            for i in obs_segs[obs].keys():
                net.fcAvg[obs, obs_segs[obs][i]['col'], obs_segs[obs][i]['row']] = obs_segs[obs][i]['sc']
