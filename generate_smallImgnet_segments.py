""" Script to generate segmented datasets of Imagenet12.
"""

# All imports

import sys
import os
import random
import numpy as np
from PIL import Image
import json
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from math import sqrt, log2
from skimage import color
from skimage.segmentation import slic, felzenszwalb

import numpy as np
import pickle
from time import time
import copy

def dataloader_small_imagenet(root, batch_size=500, image_size=224, num_workers=0, valid_size=0.1):
    """This dataloader is used only for creating segmented images.
    """

    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transform_train
    transform_val = transform_train

    trainset = datasets.ImageFolder(root + 'train/', transform=transform_train)
    valset = datasets.ImageFolder(root + 'train/', transform=transform_val)
    testset = datasets.ImageFolder(root + 'val/', transform=transform_test)

    # Splitting for val

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Load the dataset (## Sampler naturally gives shuffle)

    tr_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                           num_workers=num_workers)

    va_loader = DataLoader(valset, batch_size=batch_size, sampler=valid_sampler,
                           num_workers=num_workers)

    te_loader = DataLoader(testset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    return tr_loader, va_loader, te_loader

def avg_seg(image,segs):
    """Smooths an image with the given segments.
    """
    image = np.transpose(np.copy(image), (1, 2, 0))
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
    return torch.tensor(np.transpose(image, (2, 0, 1)))

def return_smoothed_images(images, nsegs, segmenter = slic):
    """ For a batch, returns smoothed images, segments, and number of
        actual segments of each image.
    """
    smoothed_image_stack = []
    segments_stack = []
    segments_number_stack = []
    for img in images:
        segments = segmenter(img.cpu().detach().numpy().astype('double').transpose(1, 2, 0), nsegs)
        segments_stack.append(torch.tensor(segments))
        segments_number_stack.append(len(np.unique(segments)))
        smoothed_image_stack.append(torch.Tensor(avg_seg(img.cpu().detach().numpy(), segments)))
    return torch.stack(smoothed_image_stack), torch.stack(segments_stack), torch.tensor(segments_number_stack)

def generate_dataset(loader, nsegs, path1, path2, path3):
    """ Takes in an standard image data loader, and saves .pt files of
        (standard images, its segments, labels) and (smoothed_images, labels). Also
        saves the actual number of segments.
    """
    standard_image_stack = torch.Tensor()
    smoothed_image_stack = torch.Tensor()
    segments_stack = torch.LongTensor()
    segments_number_stack = torch.LongTensor()
    labels_stack = torch.LongTensor()

    for i, data in enumerate(loader):
        images, labels = data
        standard_image_stack = torch.cat((standard_image_stack, images))
        labels_stack = torch.cat((labels_stack, labels))

        smoothed_images, segments, segments_number = return_smoothed_images(images, nsegs, segmenter=slic)
        smoothed_image_stack = torch.cat((smoothed_image_stack, smoothed_images))
        segments_stack = torch.cat((segments_stack, segments))
        segments_number_stack = torch.cat((segments_number_stack, segments_number))

    torch.save((standard_image_stack, segments_stack, labels_stack), path1)
    torch.save((smoothed_image_stack, labels_stack), path2)
    torch.save(segments_number_stack, path3)

# Get directories ready

directory = "/cluster/project/infk/krause/pmayilvahana/datasets/small_imagenet/"
directory_save = "/cluster/project/infk/krause/pmayilvahana/datasets/small_imagenet_pt/"
if not os.path.exists(directory_save):
    os.makedirs(directory_save)

tr_loader, va_loader, te_loader = dataloader_small_imagenet(directory)

# Create and save all datasets required
for nsegs in [50, 100, 500, 1000, 2500, 5000]:
    generate_dataset(tr_loader, nsegs, directory_save+str(nsegs)+"_bulk_tr.pt", directory_save+str(nsegs)+"_tr.pt", directory_save+str(nsegs)+"_numsegs_tr.pt")
    generate_dataset(va_loader, nsegs, directory_save+str(nsegs)+"_bulk_va.pt", directory_save+str(nsegs)+"_va.pt", directory_save+str(nsegs)+"_numsegs_va.pt")
    generate_dataset(te_loader, nsegs, directory_save+str(nsegs)+"_bulk_te.pt", directory_save+str(nsegs)+"_te.pt", directory_save+str(nsegs)+"_numsegs_te.pt")
