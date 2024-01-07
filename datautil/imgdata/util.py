# coding=utf-8
from torchvision import transforms
from torchvision.transforms import autoaugment, transforms

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_train(dataset, resize_size=256, crop_size=224, policy='default'):
    if dataset == 'dg5':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset == 'ColorMNIST':
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if policy == 'standard':
        # standard ImageNet policy
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            normalize
        ])
    elif policy == 'autoaugment':
        # auto augmentation policy (ImageNet)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            autoaugment.AutoAugment(),
            transforms.ToTensor(),
            normalize
        ])
    elif policy == 'randaugment':
        # random augmentation policy
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            autoaugment.RandAugment(),
            transforms.ToTensor(),
            normalize
        ])
    elif policy == 'default':
        # default policy
        transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise NotImplementedError
    return transform


def image_test(dataset, resize_size=256, crop_size=224):
    if dataset == 'dg5':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset == 'ColorMNIST':
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
