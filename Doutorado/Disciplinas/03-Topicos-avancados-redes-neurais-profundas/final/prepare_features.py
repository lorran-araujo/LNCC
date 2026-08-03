import argparse

import torch

from src.dataset import create_data_splits
from src.features import fit_and_save_pca_versions, pca_dims_for_config, save_raw_features
from src.model import build_backbone
from src.utils import load_config, set_seed


def _device_from_config(config):
    device_name = config["training"].get("device", "auto")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _prepare_split(config, device):
    data = create_data_splits(config)
    backbone, feature_dim = build_backbone(config["model"])
    backbone = backbone.to(device)

    loaders = {
        "train": data["train_loader"],
        "val": data["val_loader"],
        "test": data["test_loader"],
    }
    save_raw_features(config, backbone, loaders, device)
    fit_and_save_pca_versions(config, pca_dims_for_config(config, feature_dim))


def main():
    parser = argparse.ArgumentParser(description="Prepare frozen features and PCA caches")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = _device_from_config(config)

    k_folds = config["dataset"].get("k_folds", 1)
    if k_folds and k_folds > 1:
        raise NotImplementedError("Feature cache preparation currently supports k_folds <= 1.")

    print(f"Using device: {device}")
    _prepare_split(config, device)
    print("Feature cache prepared successfully.")


if __name__ == "__main__":
    main()
