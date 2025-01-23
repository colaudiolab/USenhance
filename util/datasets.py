# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
import torch
import torch.utils.data as data
import json
import numpy as np
from PIL import Image

class Dataset(data.Dataset):
    def __init__(
            self,
            transform=None,
            setname="train",
            json_path=""
    ):
        self.setname = setname
        self.json_path = json_path
        self.samples = self.find_images_and_targets()
        self.transform = transform

    def find_images_and_targets(self):
        images_and_targets = []
        with open(self.json_path, "r") as jf:
            imgs = json.load(jf)[self.setname]
            for k, v in imgs.items():
                images_and_targets.append([k, v])
        return images_and_targets

    def __getitem__(self, index):
        path, target = self.samples[index]

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)


def build_datasetv2(setname, args):
    if setname == "train": is_train = True
    else: is_train = False
    transform = build_transform(is_train, args)

    dataset = Dataset(transform=transform, setname=setname, json_path=args.data_path)
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset



def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
