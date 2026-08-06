import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

from utils.dataset_tiny_imagenet import TinyImageNet


def _tiny_root(dataset_path: str, name: str) -> str:
    """Return a Tiny-ImageNet root under the user-provided dataset directory."""
    return os.path.join(dataset_path, name)


def _imagenette_root(dataset_path: str) -> str:
    """Resolve the Imagenette directory while supporting common folder names."""
    candidates = [
        os.path.join(dataset_path, "imagenette2"),
        os.path.join(dataset_path, "imagenette2-320"),
        os.path.join(dataset_path, "imagenette2-160"),
        os.path.join(dataset_path, "imagenette"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def GetDataLoader(
    dataset,
    train_bs,
    test_bs,
    dataset_path="./datasets/",
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
):
    """Build train/test dataloaders used by the TaFD experiments."""
    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

    if dataset == "Tiny":
        train_dataset = TinyImageNet(
            root=_tiny_root(dataset_path, "tiny-imagenet-200"),
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandAugment(num_ops=2, magnitude=9),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = TinyImageNet(
            root=_tiny_root(dataset_path, "tiny-imagenet-200"),
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    elif dataset == "Tiny_32_200class":
        train_dataset = TinyImageNet(
            root=_tiny_root(dataset_path, "tiny-imagenet-200-32x32"),
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = TinyImageNet(
            root=_tiny_root(dataset_path, "tiny-imagenet-200-32x32"),
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    elif dataset == "Tiny_32_10class":
        root = _tiny_root(dataset_path, "tiny-imagenet_10class_32")
        train_dataset = datasets.ImageFolder(
            root=os.path.join(root, "train"),
            transform=transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(root, "val"),
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = datasets.CIFAR10(
            dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    elif dataset == "CIFAR100":
        cifar100_root = os.path.join(dataset_path, "cifar100")
        train_dataset = datasets.CIFAR100(
            cifar100_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = datasets.CIFAR100(
            cifar100_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    elif dataset == "Imagenette":
        root = _imagenette_root(dataset_path)
        train_dataset = datasets.ImageFolder(
            root=os.path.join(root, "train"),
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(root, "val"),
            transform=transforms.Compose(
                [
                    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    effective_num_workers = max(int(num_workers), 0)
    loader_kwargs = {
        "num_workers": effective_num_workers,
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
    }
    if effective_num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader
