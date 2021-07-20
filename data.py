## data.py Imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.models as models

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np



class FileDataset(Dataset):
    def __init__(self, root, transform, seed_transform=False):
        self.images, self.labels = torch.load(root)
        self.transform = transform
        self.seed_transform = seed_transform

    def __getitem__(self, idx):
        if self.seed_transform:
            random.seed(3)
            torch.manual_seed(3)
        return self.transform(self.images[idx]), self.labels[idx]

    def __len__(self):
        return len(self.images)


def dataloader_CIFAR10(args):
    """ Needs heavy rework
    """
    batch_size = args['batch_size']
    num_workers = args['num_workers']
    root = args['data_root']
    valid_size = args['val_split_size']

    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_val = transform_test

    ## Downloading and loading the dataset

    trainset = datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform_train)

    valset = datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform_val)

    testset = datasets.CIFAR10(root=root, train=False,
                                          download=True, transform=transform_test)

    ## Splitting for val

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    ## Load the dataset (## Sampler naturally gives shuffle)

    tr_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                              num_workers=num_workers)

    va_loader = DataLoader(valset, batch_size=batch_size, sampler=valid_sampler,
                                              num_workers=num_workers)

    te_loader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    
    return tr_loader, va_loader, te_loader



## Imagenet12 dataloader
def dataloader_IMAGENET12(args):
    """ Imagenet12 dataloader that returns loaders depending on 'train_type' and 'mode'.
        For

    """
    batch_size = args['batch_size']
    image_size = args['image_size'] ## model input size and not image size
    num_workers = args['num_workers']
    root = args['data_root']
    valid_size = args['val_split_size']
    mode = args['mode']
    train_type = args['train_type']
    number_of_segments = args['number_of_segments']

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    transform_val = transform_test

    if mode=='eval':
        transform_train = transform_test

    # if max number of segments then even if train_type is 'segmenter' or 'pgd_and..' then execute first if.
    if train_type=='pgd' or number_of_segments==int(224*224):
        trainset = datasets.ImageFolder(root + 'train/', transform=transform_train)
        valset = datasets.ImageFolder(root + 'train/', transform=transform_val)
        testset = datasets.ImageFolder(root + 'val/', transform=transform_test)

        ## Splitting for val

        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        ## Load the dataset (## Sampler naturally gives shuffle)

        tr_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                               num_workers=num_workers)

        va_loader = DataLoader(valset, batch_size=batch_size, sampler=valid_sampler,
                               num_workers=num_workers)

        te_loader = DataLoader(testset, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)

    elif train_type=='segmenter':
        if mode=='train':
            filename_phrase = "_"
        elif mode=='eval':
            filename_phrase = "_bulk_"
        trainset = FileDataset(root + str(number_of_segments) + filename_phrase + "tr.pt", transform_train, seed_transform)
        valset = FileDataset(root + str(number_of_segments) + filename_phrase + "va.pt", transform_valid, seed_transform)
        testset = FileDataset(root + str(number_of_segments) + filename_phrase + "te.pt", transform_test, seed_transform)

        tr_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    elif train_type=='pgd_and_segmenter' and number_of_segments:
        trainset = FileDataset(root + str(number_of_segments) + "_bulk_tr.pt", transform_train, seed_transform)
        valset = FileDataset(root + str(number_of_segments) + "_bulk_va.pt", transform_valid, seed_transform)
        testset = FileDataset(root + str(number_of_segments) + "_bulk_te.pt", transform_test, seed_transform)

        tr_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    else:
        raise ValueError(f"Invalid train type and number of segments")

    return tr_loader, va_loader, te_loader
