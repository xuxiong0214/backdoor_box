import csv
import os
import config
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import kornia.augmentation as A
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from load_data.cifar import cifar
from load_data.imagenet import imagenet
from load_data.gtsrb import gtsrb
import random

class ColorDepthShrinking(object):
    def __init__(self, c=3):
        self.t = 1 << int(8 - c)

    def __call__(self, img):
        im = np.asarray(img)
        im = (im / self.t).astype("uint8") * self.t
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(t={})".format(self.t)


class Smoothing(object):
    def __init__(self, k=3):
        self.k = k

    def __call__(self, img):
        im = np.asarray(img)
        im = cv2.GaussianBlur(im, (self.k, self.k), 0)
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(k={})".format(self.k)


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x
        
class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def get_transform(opt, train=True, c=0, k=0):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))

    if train:
        transforms_list.append(transforms.RandomCrop(size=32, padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        
    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb":
        pass
    elif opt.dataset == "imagenet":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)

    
def get_dataloader(opt, train=True, c=0, k=0):
    transform = get_transform(opt, train, c=c, k=k)
    
    if opt.dataset == "cifar10":
        if train == True:
            dataset = cifar.CIFAR(path=opt.data_root, train=True, train_type=1, tf=transform)
        else:
            dataset = cifar.CIFAR(path=opt.data_root, train=False, tf=transform)

    elif opt.dataset == "imagenet":
        if train==True:
            train_tf = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.RandomCrop(size=224, padding=4),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            dataset = imagenet.ImageNet(path="./datasets", train="poison", tf=train_tf)
        else:
            test_tf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            dataset = imagenet.ImageNet(path="./datasets", train="test", tf=test_tf)
    
    elif opt.dataset == "gtsrb":
        if train==True:
            train_tf = transforms.Compose([transforms.Resize((40, 40)),
                                           transforms.RandomCrop(size=32, padding=4),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0, 0, 0], [1, 1, 1])])
            dataset = gtsrb.GTSRB(path=opt.data_root, train=True, train_type=1, tf=train_tf)

        else:
            test_tf = transforms.Compose([transforms.Resize((40, 40)),
                                          transforms.CenterCrop(size=32),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0, 0, 0], [1, 1, 1])])
            dataset = gtsrb.GTSRB(path=opt.data_root, train=False, tf=test_tf)

    else:
        raise Exception("Invalid dataset")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True
    )
    return dataloader, transform
