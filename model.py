# model.py Imports
import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np

def get_model(args):
    """ Returns the model we want to train on
    """

    model_name = args['model_name']
    train_all_params = args['train_all_params']
    num_classes = args['num_classes'] ## this needs fix
    pretrained = args['pretrained']
    
    if model_name =='resnet18':
        model = models.resnet18(pretrained=pretrained)

        if num_classes != 1000:
            model.fc = nn.Linear(512, num_classes, bias = True)
        print(f"Loaded resnet18 with classes = {num_classes}")

        # Finetuning only the classifier
        if not train_all_params:
            for param in model.parameters():
                param.requires_grad = False

            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
            print("Training only the classifier")

    elif model_name =='resnet152':
        model = models.resnet152(pretrained=pretrained)

        if num_classes != 1000:
            model.fc = nn.Linear(2048, num_classes, bias = True)
        print(f"Loaded resnet152 with classes = {num_classes}")

        # Finetuning only the classifier
        if not train_all_params:
            for param in model.parameters():
                param.requires_grad = False

            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
            print("Training only the classifier")

    elif model_name == 'efficientnet':
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=pretrained, num_classes=num_classes)
        print(f"Loaded efficientnet with classes = {num_classes}")

        # Finetuning only the classifier
        if not train_all_params:
            for param in model.parameters():
                param.requires_grad = False

            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True
            print("Training only the classifier")
    else:
        raise ValueError(f"Given model (model key = {model_name} doesn't exist")
    return model




class wrap_model(nn.Module):
    """ Simply wraps a standard model to the smoother.
    """
    def __init__(self, model):
        super(wrap_model, self).__init__()
        self.model = model
        self.x_segs = None

    def forward(self, x):
        x = smooth_images(x, self.x_segs)
        x = self.model(x)
        return x

    def check(self, x):
        return smooth_images(x, self.x_segs)