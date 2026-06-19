from __future__ import annotations

from dataclasses import dataclass


import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetInfo:
    num_classes: int
    input_size: int
    class_names: list[str]


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    dataset_config = config["dataset"]
    name = dataset_config.get("name", "CIFAR10").lower()
    root = dataset_config.get("root", "data")
    image_size = int(dataset_config.get("image_size", 224))
    batch_size = int(dataset_config.get("batch_size", 64))
    num_workers = int(dataset_config.get("num_workers", 4))
    val_split = float(dataset_config.get("val_split", 0.1))

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )

    if name == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        eval_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=eval_transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)
        num_classes = 10
        class_names = train_dataset.classes
    elif name == "cifar100":
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        eval_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=eval_transform)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)
        num_classes = 100
        class_names = train_dataset.classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_config.get('name')}")

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    seed = int(config.get("seed", 42))
    train_subset, _ = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    _, val_subset = random_split(
        eval_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, DatasetInfo(
        num_classes=num_classes,
        input_size=image_size,
        class_names=class_names,
    )
