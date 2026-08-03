from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _build_transforms(dataset_config):
    image_size = dataset_config["image_size"]
    train_transforms = [transforms.Resize((image_size, image_size))]
    crop_padding = dataset_config.get("random_crop_padding", 0)
    color_jitter = dataset_config.get("color_jitter", 0.0)

    if crop_padding:
        train_transforms.append(transforms.RandomCrop(image_size, padding=crop_padding))

    train_transforms.append(transforms.RandomHorizontalFlip())

    if color_jitter:
        train_transforms.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
        )

    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_transform = transforms.Compose(train_transforms)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def _targets(dataset):
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "samples"):
        return np.array([sample[1] for sample in dataset.samples])
    raise ValueError("Could not infer class labels from dataset.")


def _load_base_datasets(config):
    dataset_config = config["dataset"]
    name = dataset_config["name"].lower()
    data_dir = Path(dataset_config["data_dir"])
    train_transform, eval_transform = _build_transforms(dataset_config)

    if name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=train_transform,
            download=dataset_config.get("download", True),
        )
        eval_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=eval_transform,
            download=dataset_config.get("download", True),
        )
        return train_dataset, eval_dataset

    if name == "imagefolder":
        train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
        eval_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transform)
        return train_dataset, eval_dataset

    raise ValueError(f"Unsupported dataset: {dataset_config['name']}")


def _make_loader(dataset, indices, batch_size, num_workers, shuffle):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _split_indices(targets, train_ratio, val_ratio, test_ratio, seed):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    all_indices = np.arange(len(targets))
    train_val_ratio = train_ratio + val_ratio
    train_val_indices, test_indices = train_test_split(
        all_indices,
        train_size=train_val_ratio,
        stratify=targets,
        random_state=seed,
    )

    train_size_inside = train_ratio / train_val_ratio
    train_indices, val_indices = train_test_split(
        train_val_indices,
        train_size=train_size_inside,
        stratify=targets[train_val_indices],
        random_state=seed,
    )
    return train_indices, val_indices, test_indices


def create_data_splits(config):
    dataset_config = config["dataset"]
    seed = config.get("seed", 42)
    batch_size = dataset_config["batch_size"]
    num_workers = dataset_config.get("num_workers", 2)

    train_dataset, eval_dataset = _load_base_datasets(config)
    targets = _targets(eval_dataset)

    train_indices, val_indices, test_indices = _split_indices(
        targets,
        dataset_config["train_ratio"],
        dataset_config["val_ratio"],
        dataset_config["test_ratio"],
        seed,
    )

    k_folds = dataset_config.get("k_folds", 1)
    if k_folds and k_folds > 1:
        train_val_indices = np.concatenate([train_indices, val_indices])
        train_val_targets = targets[train_val_indices]
        splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        folds = []

        for fold_train_pos, fold_val_pos in splitter.split(train_val_indices, train_val_targets):
            fold_train_indices = train_val_indices[fold_train_pos]
            fold_val_indices = train_val_indices[fold_val_pos]
            folds.append(
                {
                    "train_loader": _make_loader(
                        train_dataset, fold_train_indices, batch_size, num_workers, True
                    ),
                    "val_loader": _make_loader(
                        eval_dataset, fold_val_indices, batch_size, num_workers, False
                    ),
                }
            )

        return {
            "folds": folds,
            "test_loader": _make_loader(
                eval_dataset, test_indices, batch_size, num_workers, False
            ),
        }

    return {
        "train_loader": _make_loader(train_dataset, train_indices, batch_size, num_workers, True),
        "val_loader": _make_loader(eval_dataset, val_indices, batch_size, num_workers, False),
        "test_loader": _make_loader(eval_dataset, test_indices, batch_size, num_workers, False),
    }
