###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. (now owned by Analog Devices Inc.)
# All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
gastro Datasets
"""
import os
import sys

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as album
import cv2

import ai8x


class Gastro(Dataset):
    """
    Gastrovision Dataset.

    Args:
    root_dir (string): Root directory of dataset where ``KWS/processed/dataset.pt``
        exist.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version.
    resize_size(int, int): Width and height of the images to be resized for the dataset.
    augment_data(bool): Flag to augment the data or not. If d_type is `test`, augmentation is
        disabled.
    """

    labels = ['esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {'esophagitis':'esophagitis', 'normal-cecum':'normal-cecum', 'normal-pylorus':'normal-pylorus', 'normal-z-line':'normal-z-line', 'polyps':'polyps', 'ulcerative-colitis':'ulcerative-colitis'}

    def __init__(self, root_dir, d_type, transform=None,
                 resize_size=(128, 128), augment_data=False):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'gastro', d_type)

        if not self.__check_gastro_data_exist():
            self.__print_download_manual()
            sys.exit("Dataset not found!!!")

        self.__get_image_paths()


        self.album_transform = None
        if d_type == 'train' and augment_data:
            self.album_transform = album.Compose([
                album.GaussNoise(var_limit=(1.0, 20.0), p=0.25),
                #album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                #album.ColorJitter(p=0.5),
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                #album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                album.RandomCrop(height=resize_size[0], width=resize_size[1]),
                #album.HorizontalFlip(p=0.5),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])
        if not augment_data or d_type == 'test':
            self.album_transform = album.Compose([
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.CenterCrop(height=resize_size[0], width=resize_size[1]),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])

        self.transform = transform

    def __check_gastro_data_exist(self):
        return os.path.isdir(self.data_dir)

    def __print_download_manual(self):
        print("******************************************")
        print("Please follow the instructions below:")
        print("Download the dataset from https://datasets.simula.no/kvasir/ ")
        print("Use the split_dataset.py to split to test(0.1) and train (0.9) "
              "Copy the test and train to /data/gastro folder.")
        print("The folder structure should be /data/gastro/test and train")
        print("and inside test and train, 6 folders with images for following 6 classes:")
        print("esophagitis, normal-cecum, normal-pylorus, normal-z-line, polyps, ulcerative-colitis")
        print("******************************************")

    def __get_image_paths(self):
        self.data_list = []

        for label in self.labels:

            image_dir = os.path.join(self.data_dir, self.label_to_folder_map[label])
            for file_name in sorted(os.listdir(image_dir)):
                file_path = os.path.join(image_dir, file_name)
                if os.path.isfile(file_path):
                    self.data_list.append((file_path, self.label_to_id_map[label]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label = torch.tensor(self.data_list[index][1], dtype=torch.int64)

        image_path = self.data_list[index][0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.album_transform:
            image = self.album_transform(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_gastro(data, load_train, load_test):
    """
    Load the Cats vs Dogs dataset.
    Returns each datasample in 128x128 size.

    Data Augmentation: Train samples are augmented randomly with
        - Additive Gaussian Noise
        - RGB Shift
        - Color Jitter
        - Shift&Scale&Rotate
        - Random Crop
        - Horizontal Flip
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
    ])

    if load_train:
        train_dataset = Gastro(root_dir=data_dir, d_type='train',
                                   transform=transform, augment_data=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = Gastro(root_dir=data_dir, d_type='test', transform=transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'gastro',
        'input': (3, 128, 128),
        'output': ('esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'),
        'loader': get_gastro,
    },
]
