import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def GetDataLoader(dataset, train_batch_size, test_batch_size,
                  dataset_path='./datasets/', num_workers=8,
                  pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """Build train and test DataLoaders for the specified dataset."""

    if dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            dataset_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
        test_dataset = datasets.CIFAR10(
            dataset_path, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))

    elif dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            os.path.join(dataset_path, 'cifar100'), train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
        test_dataset = datasets.CIFAR100(
            os.path.join(dataset_path, 'cifar100'), train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))

    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose from: CIFAR10, CIFAR100")

    effective_num_workers = max(int(num_workers), 0)
    loader_kwargs = {
        'num_workers': effective_num_workers,
        'pin_memory': bool(pin_memory and torch.cuda.is_available()),
    }
    if effective_num_workers > 0:
        loader_kwargs['persistent_workers'] = bool(persistent_workers)
        loader_kwargs['prefetch_factor'] = max(int(prefetch_factor), 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        **loader_kwargs
    )

    return train_loader, test_loader
