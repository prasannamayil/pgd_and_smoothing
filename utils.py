## utils.py Imports
import torch
import numpy as np
import pickle
import re
from math import sqrt, log2
from skimage import color
from skimage.segmentation import slic, felzenszwalb ## Felzenszwalb also added
import torch
import collections

## PGD imports

import sys
import os
import random
import numpy as np
from PIL import Image
import json
import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from skimage import color
from skimage.segmentation import slic, felzenszwalb

from time import time
import sys
import numpy as np
import pickle

# Function that smooths images based on segments (rewrite efficiently later)

def avg_seg(image,segs):
    image = np.transpose(np.copy(image),(1,2,0))
    size = image.shape[0]
    sums = {}
    ns = {}
    for x in range(size):
        for y in range(size):
            zone = segs[x][y]
            if not zone in sums:
                sums[zone] = np.zeros_like(image[0][0])
                ns[zone] = 0
            sums[zone] += image[x][y]
            ns[zone] += 1
    for x in range(size):
        for y in range(size):
            zone = segs[x][y]
            image[x][y] = sums[zone]/ns[zone]
    return torch.tensor(np.transpose(image,(2,0,1)))

# Function that I created. Does what batch_seg does but for single image.It is used by square circle dataset
def singleSeg(img, nsegs, seg):
    return avg_seg(img.cpu(),seg(img.cpu(),nsegs))


# Segments a batch of images and returns a smoothed version
def batchSeg(images,nsegs,seg):
    tab = [avg_seg(img.cpu(), seg(img.cpu(), nsegs)) for img in images]
    return torch.stack(tab)

# SLIC function to segment image
def seg_SLIC(image,nsegs, compactness = 10.0, max_iter = 10):
    im = image.detach().numpy().astype('double').transpose(1,2,0)
    return slic(im,n_segments=nsegs, compactness = compactness, max_iter = max_iter)


def setup_device(required_gpus):
    actual_gpus = torch.cuda.device_count()
    if required_gpus > 0 and actual_gpus == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        required_gpus = 0
    if required_gpus > actual_gpus:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(required_gpus, actual_gpus))
        required_gpus = actual_gpus
    device = torch.device('cuda:0' if required_gpus > 0 else 'cpu')
    list_ids = list(range(required_gpus))
    return device, list_ids

# Saves entire state of training with state_dict, optimizer, scheduler etc.

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'best.pth')
        torch.save(state, filename)

# Function that computes the expected gradient norm of the input for a batch.

def l2_norm_grads(images, labels, net, criterion, optimizer, return_input_grad=False):
    images.requires_grad = True # Setting true so that gradients can be backproped to input
    temp_loss = criterion(net(images), labels)
    temp_loss.backward()

    input_grad = images.grad

    optimizer.zero_grad() # Make grads zero
    images.requires_grad = False
    
    expected_norm_grads = input_grad.pow(2).sum(dim=(1, 2, 3)).pow(0.5).mean().item()
    if return_input_grad:
        return input_grad.cpu().detach().clone().numpy(), expected_norm_grads
    else:
        return expected_norm_grads

# The dictionary of all losses, accuracies, gradient norms for natural images and corresponding adversaries.

def init_stat_dictionaries(args):
    if not args['resume_training']:
        losses_adversaries = {
                  'tr_epoch': [],
                  'va_epoch': [],
                  'te_epoch': []
        }


        vulnerabilities = {
                          'tr_epoch': [],
                          'va_epoch': [],
                          'te_epoch': []
        }

        accuracies_adversaries = {
                      'tr_epoch': [],
                      'va_epoch': [],
                      'te_epoch': []
        }

        grad_norms_adversaries = {
                      'tr_epoch': [],
                      'va_epoch': [],
                      'te_epoch': []
        }

        grad_norms_input = {
                      'tr_epoch': [],
                      'va_epoch': [],
                      'te_epoch': []
        }
    else:
          with open(args['directory']+'losses_adversaries.pickle', 'rb') as handle:
                losses_adversaries = pickle.load(handle)
          
          with open(args['directory']+'vulnerabilities.pickle', 'rb') as handle:
                vulnerabilities = pickle.load(handle)
          
          with open(args['directory']+'accuracies_adversaries.pickle', 'rb') as handle:
                accuracies_adversaries = pickle.load(handle)
          
          with open(args['directory']+'grad_norms_adversaries.pickle', 'rb') as handle:
                grad_norms_adversaries = pickle.load(handle)

          with open(args['directory']+'grad_norms_input.pickle', 'rb') as handle:
                grad_norms_input = pickle.load(handle)

    return losses_adversaries, vulnerabilities, accuracies_adversaries, \
            grad_norms_adversaries, grad_norms_input


# Function to save all trains statistics

def save_dictionaries(directory, vulnerabilities, losses_adversaries, accuracies_adversaries, grad_norms_input, grad_norms_adversaries):
  with open(directory+'vulnerabilities.pickle', 'wb') as handle:
      pickle.dump(vulnerabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'losses_adversaries.pickle', 'wb') as handle:
      pickle.dump(losses_adversaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'accuracies_adversaries.pickle', 'wb') as handle:
      pickle.dump(accuracies_adversaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'grad_norms_input.pickle', 'wb') as handle:
      pickle.dump(grad_norms_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'grad_norms_adversaries.pickle', 'wb') as handle:
      pickle.dump(grad_norms_adversaries, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Eval utils functions

def init_eval_stat_dictionaries(eval_attack_keys, eval_data_keys):
    ## Getting the stat dicts ready
    adversaries_images = dict({i:{j: None for j in eval_data_keys} for i in eval_attack_keys})
    normal_images_dict = dict({i:{j: None for j in eval_data_keys} for i in eval_attack_keys})

    accuracies_adversaries = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    vulnerabilities = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})

    adv_norm_grads = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    logit_images = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    correct_labels = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    input_grad_norms_stack = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    input_grad_stack = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    logit_advs = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})
    advs_success_failures = dict({i:{j: {} for j in eval_data_keys} for i in eval_attack_keys})

    return adversaries_images, normal_images_dict, accuracies_adversaries, vulnerabilities, \
      adv_norm_grads, logit_images, correct_labels, input_grad_norms_stack, input_grad_stack, \
      logit_advs, advs_success_failures, wts_change_dict, wts_q_norm, wts_k_norm 


# Function to save all eval statistics

def save_eval_dictionaries(directory, vulnerabilities, accuracies_adversaries, adversaries_images, normal_images_dict, adv_norm_grads, \
                           logit_images, correct_labels, input_grad_norms_stack, input_grad_stack, logit_advs, advs_success_failures):
  with open(directory+'vulnerabilities.pickle', 'wb') as handle:
      pickle.dump(vulnerabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'accuracies_adversaries.pickle', 'wb') as handle:
      pickle.dump(accuracies_adversaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'adversaries_images.pickle', 'wb') as handle:
      pickle.dump(adversaries_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'normal_images.pickle', 'wb') as handle:
      pickle.dump(normal_images_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'adv_norm_grads.pickle', 'wb') as handle:
      pickle.dump(adv_norm_grads, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'logit_images.pickle', 'wb') as handle:
      pickle.dump(logit_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'correct_labels.pickle', 'wb') as handle:
      pickle.dump(correct_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'input_grad_norms_stack.pickle', 'wb') as handle:
      pickle.dump(input_grad_norms_stack, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'input_grad_stack.pickle', 'wb') as handle:
      pickle.dump(input_grad_stack, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'logit_advs.pickle', 'wb') as handle:
      pickle.dump(logit_advs, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open(directory+'advs_success_failures.pickle', 'wb') as handle:
    pickle.dump(advs_success_failures, handle, protocol=pickle.HIGHEST_PROTOCOL)


## SLIC function to segment image

def return_segments(images, nsegs, segmenter='slic', **kwargs):
    ''' Takes in a image tensor batch of type cpu and outputs a
    tensor batch of segmented images. We use two segmenters: slic,
    and felzenszwalb (needs to be written).
    '''
    images_segments = []
    for image in images:
        image = image.detach().numpy().astype('double').transpose(1, 2, 0)

        if segmenter=='slic':
            compactness = kwargs.pop('compactness', 10.0)
            max_iter = kwargs.pop('max_num_iter', 10)
            images_segments.append(
                torch.tensor(slic(image, n_segments=nsegs, compactness=compactness, max_iter=max_iter)))

        else:
            raise ValueError(f"Only slic available at this moment")

    return torch.stack(images_segments)

def smooth_images(x,x_segs):
    ''' Takes unsmoothed image, and its segments as input and
    returns smoothed image. Requires presegmented images to be stored.
    '''
    y = torch.zeros_like(x)
    for obs in range(len(x)):
        segs = x_segs[obs, :].detach().numpy()
        for seg in np.unique(segs):
            locs = (segs == seg).nonzero()
            y[obs, :, locs[0], locs[1]] = x[obs, :, locs[0], locs[1]].mean(-1).unsqueeze(1)
    return y