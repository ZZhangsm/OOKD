# coding=utf-8
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from datautil.util import Nmax
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder, MNIST
from torchvision.datasets.folder import default_loader

def color_dataset(images, labels, environment):
    # # Subsample 2x for computational convenience
    # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit
    labels = (labels < 5).float()
    # Flip label with probability 0.25
    labels = torch_xor_(labels, torch_bernoulli_(0.25, len(labels)))

    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor_(labels, torch_bernoulli_(environment,
                                                   len(labels)))
    images = torch.stack([images, images], dim=1)
    # Apply the color to the image by zeroing out the other color channel
    images[torch.tensor(range(len(images))), (
        1 - colors).long(), :, :] *= 0

    x = images.float().div_(255.0)
    y = labels.view(-1).long()

    return TensorDataset(x, y)

def torch_bernoulli_(p, size):
    return (torch.rand(size) < p).float()

def torch_xor_(a, b):
    return (a - b).abs()

class ImageDataset(object):
    """
    Image dataset class
    """
    def __init__(self, dataset, task, root_dir, domain_name,
                 domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default'):
        self.dataset = dataset
        self.domain_num = 0
        self.task = task
        if root_dir is None:
            raise ValueError('Data directory not specified!')

        if self.dataset == 'CelebA':
            file_names = []
            attributes = []
            target_attribute_id = 9
            split_csv = Path(root_dir, 'blond_split', f'{domain_name}.csv')
            with open(split_csv) as f:
                reader = csv.reader(f)
                next(reader)  # discard header
                for row in reader:
                    file_names.append(row[0])
                    attributes.append(np.array(row[1:], dtype=int))
            attributes = np.stack(attributes, axis=0)
            self.imgs = list(zip(file_names, list(attributes[:, target_attribute_id])))
            imgs = [root_dir + 'img_align_celeba/' + item[0] for item in self.imgs]
            # self.root_dir = Path(root_dir, 'img_align_celeba')
        elif self.dataset == 'ColorMNIST':
            # ENV_LEN = 3
            ENV = [0.1, 0.2, 0.9]
            original_dataset_tr = MNIST(root_dir, train=True, download=True)
            original_dataset_te = MNIST(root_dir, train=False, download=True)
            original_images = torch.cat((original_dataset_tr.data,
                                         original_dataset_te.data))

            original_labels = torch.cat((original_dataset_tr.targets,
                                         original_dataset_te.targets))
            shuffle = torch.randperm(len(original_images))

            original_images = original_images[shuffle]
            original_labels = original_labels[shuffle]
            images = original_images[domain_label::len(ENV)]
            labels = original_labels[domain_label::len(ENV)]
            self.imgs = color_dataset(images, labels, ENV[domain_label])
            imgs = [item[0] for item in self.imgs]

        else:
            self.imgs = ImageFolder(root_dir+domain_name).imgs
            imgs = [item[0] for item in self.imgs]

        labels = [item[1] for item in self.imgs]
        self.labels = np.array(labels)
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform

        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices

        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * \
            (domain_label-Nmax(test_envs, domain_label))


    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        if self.dataset == 'ColorMNIST':
            index = self.indices[index]
            img = self.x[index]
            ctarget = self.labels[index]
            dtarget = self.dlabels[index]
        else:
            index = self.indices[index]
            img = self.input_trans(self.loader(self.x[index]))
            ctarget = self.target_trans(self.labels[index])
            dtarget = self.target_trans(self.dlabels[index])
        return img, ctarget, dtarget

    def __len__(self):
        return len(self.indices)

    def dataloader(self, batch_size=32, shuffle=False, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class ImageDatasetSample(object):
    """
    Sampled Image Dataset
    """
    def __init__(self, dataset, task, root_dir, domain_name,
                 domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default',
                 k=4096, sample_mode='exact', is_sample=True, percent=1.0):
        self.dataset = dataset
        # self.imgs = ImageFolder(root_dir+domain_name).imgs
        self.domain_num = 0
        self.task = task

        if self.dataset == 'CelebA':
            file_names = []
            attributes = []
            target_attribute_id = 9
            split_csv = Path(root_dir, 'blond_split', f'{domain_name}.csv')
            with open(split_csv) as f:
                reader = csv.reader(f)
                next(reader)  # discard header
                for row in reader:
                    file_names.append(row[0])
                    attributes.append(np.array(row[1:], dtype=int))
            attributes = np.stack(attributes, axis=0)
            self.imgs = list(zip(file_names, list(attributes[:, target_attribute_id])))
            imgs = [root_dir + 'img_align_celeba/' + item[0] for item in self.imgs]
            # self.root_dir = Path(root_dir, 'img_align_celeba')
        elif self.dataset == 'ColorMNIST':
            # ENV_LEN = 3
            ENV = [0.1, 0.2, 0.9]
            original_dataset_tr = MNIST(root_dir, train=True, download=True)
            original_dataset_te = MNIST(root_dir, train=False, download=True)
            original_images = torch.cat((original_dataset_tr.data,
                                         original_dataset_te.data))

            original_labels = torch.cat((original_dataset_tr.targets,
                                         original_dataset_te.targets))
            shuffle = torch.randperm(len(original_images))

            original_images = original_images[shuffle]
            original_labels = original_labels[shuffle]
            images = original_images[domain_label::len(ENV)]
            labels = original_labels[domain_label::len(ENV)]
            self.imgs = color_dataset(images, labels, ENV[domain_label])
            imgs = [item[0] for item in self.imgs]
        else:
            self.imgs = ImageFolder(root_dir+domain_name).imgs
            imgs = [item[0] for item in self.imgs]

        # imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.labels = np.array(labels)
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * \
            (domain_label-Nmax(test_envs, domain_label))

        self.k = k # Number of negative samples
        self.is_sample = is_sample # Whether to sample contrastive examples or not
        self.percent = percent # Percentage of samples to be used for training
        self.sample_mode = sample_mode # Sampling mode ('exact' or 'relax')
        num_classes = len(np.unique(self.labels))
        num_samples = len(self.labels)

        self.cls_pos = [[] for _ in range(num_classes)]
        self.cls_neg = [[] for _ in range(num_classes)]

        for i in range(num_samples):
            self.cls_pos[self.labels[i]].append(i)

        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    self.cls_neg[i].extend(self.cls_pos[j])

        self.cls_pos = [np.asarray(self.cls_pos[i]) for i in range(num_classes)]
        self.cls_neg = [np.asarray(self.cls_neg[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_neg[0]) * percent)
            self.cls_neg = [np.random.permutation(self.cls_neg[i])[:n] for i in range(num_classes)]

        # self.cls_pos = np.asarray(self.cls_pos, dtype=object)
        # self.cls_neg = np.asarray(self.cls_neg, dtype=object)

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        if self.dataset in ['ColorMNIST']:
            index = self.indices[index]
            img = self.x[index]
            ctarget = self.labels[index]
            dtarget = self.dlabels[index]
        else:
            index = self.indices[index]
            img = self.input_trans(self.loader(self.x[index]))
            ctarget = self.target_trans(self.labels[index])
            dtarget = self.target_trans(self.dlabels[index])

        if not self.is_sample:
            return img, ctarget, dtarget
        else:
            if self.sample_mode == 'exact':
                pos_idx = index
            elif self.sample_mode == 'relax':
                pos_idx = np.random.choice(self.cls_pos[ctarget], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.sample_mode)

            replace = True if self.k > len(self.cls_neg[ctarget]) else False
            # neg_idx = np.random.choice(self.cls_neg[ctarget], self.k, replace=replace).astype(int)
            neg_idx = np.random.choice(self.cls_neg[ctarget], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, ctarget, dtarget, index, sample_idx

    def __len__(self):
        return len(self.indices)

    def dataloader(self, batch_size=32, shuffle=False, num_workers=4):
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
