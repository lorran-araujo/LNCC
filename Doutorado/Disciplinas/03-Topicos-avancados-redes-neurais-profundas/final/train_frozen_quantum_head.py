import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DenseHeadClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.head(features)


def _quantum_forward(quantum_model, encoded_features):
    try:
        quantum_outputs = quantum_model.quantum_layer(encoded_features)
        if quantum_outputs.ndim == 1:
            quantum_outputs = quantum_outputs.unsqueeze(0)
        return quantum_outputs
    except RuntimeError as error:
        if encoded_features.ndim == 1:
            raise error
        return torch.stack(
            [quantum_model.quantum_layer(sample) for sample in encoded_features], dim=0
        )


def _precompute_quantum_outputs(quantum_model, loader, device):
    quantum_model.eval()
    outputs = []
    labels = []

    with torch.no_grad():
        for features, batch_labels in loader:
            features = features.to(device)
            encoded_features = quantum_model._prepare_inputs(features)
            batch_outputs = _quantum_forward(quantum_model, encoded_features)
            outputs.append(batch_outputs.detach().cpu().float())
            labels.append(batch_labels.detach().cpu().long())

    return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)


def _precompute_quantum_loaders(quantum_model, feature_loaders, quantum_device, head_batch_size):
    loaders = {}
    for split in ("train", "val", "test"):
        features, labels = _precompute_quantum_outputs(
            quantum_model,
            feature_loaders[f"{split}_loader"],
            quantum_device,
        )
        loaders[f"{split}_loader"] = _make_feature_loader(
            features,
            labels,
            batch_size=head_batch_size,
            shuffle=(split == "train"),
        )

    return loaders


def _count_trainable_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _force_quantum_amplitude_6q(config):
    config["dataset"]["k_folds"] = 1
    config["model"]["classifier"] = "quantum"
    config["model"].setdefault("quantum", {})
    config["model"]["quantum"]["encoding"] = "amplitude"
    config["model"]["quantum"]["num_qubits"] = 6
    return config


def _device_from_config(config):
    device_name = config["training"].get("device", "auto")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _feature_root(config):
    dataset_config = config["dataset"]
    model_config = config["model"]
    feature_config = config.get("features", {})
    cache_dir = Path(feature_config.get("cache_dir", "./features"))
    dataset_name = dataset_config["name"].lower()
    backbone = model_config.get("backbone", "resnet18").lower()
    seed = config.get("seed", 42)
    return cache_dir / dataset_name / backbone / f"seed_{seed}" / "split"


def _pca_feature_path(config, split, dim):
    return _feature_root(config) / "pca" / f"dim_{dim}" / f"{split}.pt"


def _load_feature_file(path):
    data = torch.load(path, map_location="cpu")
    return data["features"].float(), data["labels"].long(), data.get("metadata", {})


def _make_feature_loader(features, labels, batch_size, shuffle):
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=shuffle)


def _load_amplitude_6q_loaders(config):
    dim = 2 ** config["model"]["quantum"]["num_qubits"]
    batch_size = config.get("features", {}).get(
        "quantum_batch_size",
        config.get("features", {}).get("batch_size", config["dataset"].get("batch_size", 64)),
    )
    loaders = {}
    feature_metadata = {"feature_source": "pca", "feature_dim": dim}

    for split in ("train", "val", "test"):
        path = _pca_feature_path(config, split, dim)
        if not path.exists():
            raise FileNotFoundError(
                f"PCA feature cache not found at {path}. Run prepare_features.py first."
            )
        features, labels, metadata = _load_feature_file(path)
        loaders[f"{split}_loader"] = _make_feature_loader(
            features, labels, batch_size=batch_size, shuffle=(split == "train")
        )
        if split == "train":
            feature_metadata["feature_dim"] = features.size(1)
            feature_metadata["metadata"] = metadata

    return loaders, feature_metadata["feature_dim"], feature_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Train a dense classical head on top of a frozen trained quantum classifier."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        default="outputs/teste3/quantum_amplitude_6q/best_model.pt",
        help="Checkpoint from the trained quantum amplitude 6-qubit model.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/teste3/quantum_amplitude_6q_frozen_vqc_head256",
        help="Directory where the new head experiment will be saved.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension of the new dense head.",
    )
    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=None,
        help="Batch size used to train the dense head after precomputing quantum outputs.",
    )
    args = parser.parse_args()

    from src.model import build_feature_classifier
    from src.trainer import Trainer
    from src.utils import load_config, save_json, set_seed

    config = load_config(args.config)
    config = _force_quantum_amplitude_6q(config)
    config["training"]["output_dir"] = args.output_dir
    set_seed(config.get("seed", 42))

    head_device = _device_from_config(config)
    quantum_device = torch.device("cpu")
    head_batch_size = args.head_batch_size or config.get("features", {}).get(
        "batch_size", config["dataset"].get("batch_size", 64)
    )

    print(f"Quantum device: {quantum_device}")
    print(f"Dense head device: {head_device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    feature_loaders, input_dim, feature_metadata = _load_amplitude_6q_loaders(config)
    quantum_model = build_feature_classifier(input_dim, config["model"]).to(quantum_device)
    quantum_model.load_state_dict(torch.load(checkpoint_path, map_location=quantum_device))
    for parameter in quantum_model.parameters():
        parameter.requires_grad = False

    print("Precomputing frozen quantum outputs...")
    data = _precompute_quantum_loaders(
        quantum_model,
        feature_loaders,
        quantum_device,
        head_batch_size=head_batch_size,
    )

    model = DenseHeadClassifier(
        input_dim=quantum_model.num_qubits,
        hidden_dim=args.hidden_dim,
        num_classes=config["model"]["num_classes"],
    ).to(head_device)

    trainer = Trainer(model, config, head_device)
    history = trainer.fit(data["train_loader"], data["val_loader"])
    test_metrics = trainer.evaluate(data["test_loader"])
    trainable_params = _count_trainable_params(model)

    save_json(
        {
            "model": {
                "base_checkpoint": str(checkpoint_path),
                "base_quantum": config["model"].get("quantum", {}),
                "head": {
                    "hidden_dim": args.hidden_dim,
                    "num_classes": config["model"]["num_classes"],
                },
            },
            "features": feature_metadata,
            "trainable_params": trainable_params,
            "history": history,
            "test": test_metrics,
        },
        Path(config["training"]["output_dir"]) / "metrics.json",
    )
    print(f"Trainable params: {trainable_params}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
