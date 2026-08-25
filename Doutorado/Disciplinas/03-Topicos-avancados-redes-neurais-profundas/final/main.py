import argparse
from pathlib import Path

import torch

from src.dataset import create_data_splits
from src.features import (
    features_exist,
    fit_and_save_pca_versions,
    load_feature_loaders,
    pca_feature_path,
    pca_features_exist,
    quantum_input_dim,
    raw_feature_path,
    save_raw_features,
)
from src.model import build_backbone, build_feature_classifier
from src.trainer import Trainer
from src.utils import load_config, save_json, set_seed


def _ensure_feature_cache(config, device):
    feature_config = config.get("features", {})
    if feature_config.get("use_cache", True) and features_exist(config):
        return

    if not feature_config.get("prepare_if_missing", False):
        missing = raw_feature_path(config, "train")
        raise FileNotFoundError(
            f"Feature cache not found at {missing}. Run "
            "`python3 prepare_features.py --config config.yaml` first or set "
            "features.prepare_if_missing: true."
        )

    data = create_data_splits(config)
    backbone, feature_dim = build_backbone(config["model"])
    backbone = backbone.to(device)
    loaders = {
        "train": data["train_loader"],
        "val": data["val_loader"],
        "test": data["test_loader"],
    }
    save_raw_features(config, backbone, loaders, device)


def _classifier_feature_source(config):
    model_config = config["model"]
    classifier_type = model_config.get("classifier", "mlp").lower()

    if classifier_type == "mlp":
        mlp_config = model_config.get("mlp", {})
        source = mlp_config.get("feature_source", "raw").lower()
        return source, mlp_config.get("reduced_dim")

    if classifier_type == "quantum":
        quantum_config = model_config.get("quantum", {})
        encoding = quantum_config.get("encoding", "angle")
        num_qubits = quantum_config.get("num_qubits", 4)
        return "pca", quantum_input_dim(encoding, num_qubits)

    raise ValueError(f"Unsupported classifier: {classifier_type}")


def _ensure_pca_cache(config, dim):
    feature_config = config.get("features", {})
    if feature_config.get("use_cache", True) and pca_features_exist(config, dim):
        return dim

    raw_features = torch.load(raw_feature_path(config, "train"), map_location="cpu")["features"]
    actual_dim = min(dim, raw_features.size(1), raw_features.size(0))
    if feature_config.get("use_cache", True) and pca_features_exist(config, actual_dim):
        return actual_dim

    if not feature_config.get("prepare_if_missing", False):
        missing = pca_feature_path(config, "train", actual_dim)
        raise FileNotFoundError(
            f"PCA feature cache not found at {missing}. Run "
            "`python3 prepare_features.py --config config.yaml` first or set "
            "features.prepare_if_missing: true."
        )

    fit_and_save_pca_versions(config, [dim])
    return actual_dim


def _load_training_loaders(config, device):
    _ensure_feature_cache(config, device)
    source, dim = _classifier_feature_source(config)
    classifier = config["model"].get("classifier", "mlp").lower()
    feature_config = config.get("features", {})
    batch_size = feature_config.get("batch_size", config["dataset"].get("batch_size", 64))

    if classifier == "quantum":
        batch_size = feature_config.get("quantum_batch_size", batch_size)

    if source == "pca":
        dim = _ensure_pca_cache(config, dim)

    loaders = load_feature_loaders(config, source, dim, batch_size=batch_size)
    input_dim = loaders["train_loader"].dataset.tensors[0].size(1)
    return loaders, input_dim, {"feature_source": source, "feature_dim": input_dim}


def _count_trainable_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_single_split(config, device):
    data, input_dim, feature_metadata = _load_training_loaders(config, device)
    model = build_feature_classifier(input_dim, config["model"]).to(device)
    trainable_params = _count_trainable_params(model)
    trainer = Trainer(model, config, device)

    history = trainer.fit(data["train_loader"], data["val_loader"])
    test_metrics = trainer.evaluate(data["test_loader"])

    output_dir = Path(config["training"]["output_dir"])
    save_json(
        {
            "model": config["model"],
            "features": feature_metadata,
            "trainable_params": trainable_params,
            "history": history,
            "test": test_metrics,
        },
        output_dir / "metrics.json",
    )
    print(f"Test metrics: {test_metrics}")


def run_cross_validation(config, device):
    raise NotImplementedError("Feature-based training currently supports k_folds <= 1.")


def _force_cpu_for_quantum(config, device):
    classifier = config["model"].get("classifier", "mlp").lower()
    if classifier == "quantum" and device.type != "cpu":
        print("Quantum classifier uses PennyLane default.qubit; switching device to CPU.")
        return torch.device("cpu")
    return device


def main():
    parser = argparse.ArgumentParser(description="Image classification training")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = config["training"].get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")
    device = _force_cpu_for_quantum(config, device)

    k_folds = config["dataset"].get("k_folds", 1)
    if k_folds and k_folds > 1:
        run_cross_validation(config, device)
    else:
        run_single_split(config, device)


if __name__ == "__main__":
    main()
