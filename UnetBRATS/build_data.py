from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import torch
import os
import random
import glob

import torch.utils.data.sampler as sampler
# import torchvision.transforms as transforms
from torchvision import transforms
# import torchvision.transforms.functional as transforms_f

from module_list import *
import h5py
import numpy

import cv2


# --------------------------------------------------------------------------------
# Define data augmentation
# --------------------------------------------------------------------------------
def transform_u(image, label, logits=None, crop_size=(112, 112, 80), scale_size=(0.8, 1.0), augmentation=True):
    raw_w, raw_h, raw_c = image.shape

    scale = random.uniform(scale_size[0], scale_size[1])
    w_, h_ = int(image.shape[0] * scale), int(image.shape[1] * scale)
    # print( w_, h_)
    image = cv2.resize(image, dsize=(h_, w_), interpolation=cv2.INTER_CUBIC)
    label = cv2.resize(label, dsize=(h_, w_), interpolation=cv2.INTER_NEAREST)
    if logits is not None:
        logits = cv2.resize(logits, dsize=(h_, w_), interpolation=cv2.INTER_NEAREST)
    # print(label.shape)

    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_w, raw_h, raw_c)

    if image.shape[0] <= crop_size[0] or image.shape[1] <= crop_size[1] or image.shape[2] <= \
            crop_size[2]:
        pw = max((crop_size[0] - image.shape[0]) // 2 + 3, 0)
        ph = max((crop_size[1] - image.shape[1]) // 2 + 3, 0)
        pd = max((crop_size[2] - image.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='reflect')
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=255)
        if logits is not None:
            logits = np.pad(logits, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape

    w1 = int(round((w - crop_size[0]) / 2.))
    h1 = int(round((h - crop_size[1]) / 2.))
    d1 = int(round((d - crop_size[2]) / 2.))

    label = label[w1:w1 + crop_size[0], h1:h1 + crop_size[1], d1:d1 + crop_size[2]]
    image = image[w1:w1 + crop_size[0], h1:h1 + crop_size[1], d1:d1 + crop_size[2]]
    if logits is not None:
        logits = logits[w1:w1 + crop_size[0], h1:h1 + crop_size[1], d1:d1 + crop_size[2]]

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            scale = random.uniform(.75, 1.25)
            image = image * scale
            image = numpy.clip(image, a_min=0., a_max=1.)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = 0.1
            mu = 0
            noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]),
                            -2 * sigma, 2 * sigma)
            noise = noise + mu
            image = image + noise
        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            if logits is not None:
                logits = np.rot90(logits, k)
                logits = np.flip(logits, axis=axis).copy()

    # Transform to tensor
    image = torch.tensor(image)
    label = torch.tensor(label).long()

    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = torch.tensor(logits)

    # Apply (ImageNet) normalisation

    image = (image - image.min()) / (image.max() - image.min())
    if logits is not None:
        return image, label, logits
    else:
        return image, label


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']
        # if label.shape[0] <= self.output_size[0]:
        #    print("pad")
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)  # why 0
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class Normalise(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label'].astype(numpy.int32)
        return {'image': (image - image.min()) / (image.max() - image.min()), 'label': label}


class RandomBrightness(object):
    def __init__(self, region):
        self.region = region

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        scale = random.uniform(self.region[0], self.region[1])
        image = image * scale
        image = numpy.clip(image, a_min=0., a_max=1.)
        sample = {'image': image, 'label': label}
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        label = sample['label'].astype(numpy.int32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}


def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform_u(data_pil, label_pil, logits_pil,
                                                      crop_size=crop_size,
                                                      scale_size=scale_size,
                                                      augmentation=apply_augmentation)

        data_list.append(aug_data.unsqueeze(0).unsqueeze(0))
        label_list.append(aug_label.unsqueeze(0))
        logits_list.append(aug_logits.unsqueeze(0))

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans


# -------------------------------------------------------------------------------
# Define indices for labelled, unlabelled training images, and test images
# --------------------------------------------------------------------------------
def get_idx(root, list_path,train=False, test_val=False, label_num=5):
    list_path=os.path.expanduser(list_path)
    if train:
        labels = []
        unlabels = []
        file_name_all = list_path + 'train.list'
        i=0
        with open(file_name_all) as f:
            files = f.readlines()
            for item in files:
                name = item.strip()
                if i < label_num:
                    labels.append(name)
                else:
                    unlabels.append(name)
                i+=1
        return labels, unlabels
    if test_val:
        file_name_val = list_path + 'val.list'
        vals = []
        with open(file_name_val) as f:
            files = f.readlines()
            for item in files:
                name = item.strip()
                vals.append(name)
        file_name_test = list_path + 'test.list'
        tests = []
        with open(file_name_test) as f:
            files = f.readlines()
            for item in files:
                name = item.strip()
                tests.append(name)
        return vals,tests
    else:
        file_name_val = list_path + 'eval.list'

        vals = []
        with open(file_name_val) as f:
            files = f.readlines()
            for item in files:
                name = item.strip()
                vals.append(name)
        return vals


# --------------------------------------------------------------------------------
# Create dataset in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataset(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, apply_partial=None, partial_seed=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.dataset = dataset
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed
        self.training_transform = transforms.Compose([
            RandomRotFlip(),
            Normalise(),
            RandomBrightness((.75, 1.25)),
            RandomCrop((96, 96, 96)),
            ToTensor(),
        ])
        self.testing_transform = transforms.Compose([
            Normalise()
        ])

    def __getitem__(self, index):
        image_name = self.idx_list[index]
        h5f = h5py.File(self.root + image_name + ".h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if not self.augmentation:
            sample = self.testing_transform(sample)
            return sample['image'], sample['label'], image_name
        sample = self.training_transform(sample)
        return sample['image'], sample['label'], image_name

    def __len__(self):
        return len(self.idx_list)


# --------------------------------------------------------------------------------
# Create data loader in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataLoader:
    def __init__(self, dataset, num_labels, base_data_path):
        self.dataset = dataset
        self.im_size = [96, 96, 96]
        self.crop_size = [96, 96, 96]
        self.num_segments = 2
        self.scale_size = (0.5, 1.5)
        self.batch_size = 2
        self.data_path = base_data_path + "/data/"
        self.data_list_path = base_data_path + "/datalist/"
        self.train_l_idx, self.train_u_idx = get_idx(self.data_path, self.data_list_path, train=True,
                                                     label_num=num_labels)
        self.val_idx,self.test_idx = get_idx(self.data_path, self.data_list_path, train=False,test_val=True)


    def build(self, supervised=False, partial=None, partial_seed=None):
        train_l_dataset = BuildDataset(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed)
        train_u_dataset = BuildDataset(self.data_path, self.dataset, self.train_u_idx,
                                       crop_size=self.crop_size, scale_size=(1.0, 1.0),
                                       augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed)
        test_dataset = BuildDataset(self.data_path, self.dataset, self.val_idx,
                                    crop_size=self.im_size, scale_size=(1.0, 1.0),
                                    augmentation=False, train=False)
        val_dataset = BuildDataset(self.data_path, self.dataset, self.val_idx,
                                   crop_size=self.im_size, scale_size=(1.0, 1.0),
                                   augmentation=False, train=False)
        if supervised:  # no unlabelled dataset needed, double batch-size to match the same number of training samples
            self.batch_size = self.batch_size * 2

        num_samples = self.batch_size * 200  # for total 40k iterations with 200 epochs

        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,
        )

        if not supervised:
            train_u_loader = torch.utils.data.DataLoader(
                train_u_dataset,
                batch_size=self.batch_size,
                sampler=sampler.RandomSampler(data_source=train_u_dataset,
                                              replacement=True,
                                              num_samples=num_samples),
                drop_last=True,
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
        )

        if supervised:
            return train_l_loader, test_loader
        else:
            return train_l_loader, train_u_loader, val_loader, test_loader

