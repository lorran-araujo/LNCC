from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


def quantum_input_dim(encoding, num_qubits):
    encoding = encoding.lower()
    if encoding in {"angle", "reupload_angle"}:
        return num_qubits
    if encoding == "dense_angle":
        return 3 * num_qubits
    if encoding == "amplitude":
        return 2 ** num_qubits
    raise ValueError(f"Unsupported encoding: {encoding}")


def pca_dims_for_config(config, feature_dim=None):
    feature_config = config.get("features", {})
    qubits = feature_config.get("qubits", [4, 6, 8])
    encodings = feature_config.get("encodings", ["angle", "amplitude"])
    dims = set()

    for num_qubits in qubits:
        for encoding in encodings:
            dim = quantum_input_dim(encoding, num_qubits)
            if feature_dim is not None:
                dim = min(dim, feature_dim)
            dims.add(dim)

    return sorted(dims)


def feature_root(config, fold_name="split"):
    dataset_config = config["dataset"]
    model_config = config["model"]
    feature_config = config.get("features", {})
    cache_dir = Path(feature_config.get("cache_dir", "./features"))
    dataset_name = dataset_config["name"].lower()
    backbone = model_config.get("backbone", "resnet18").lower()
    seed = config.get("seed", 42)
    return cache_dir / dataset_name / backbone / f"seed_{seed}" / fold_name


def raw_feature_path(config, split, fold_name="split"):
    return feature_root(config, fold_name) / "raw" / f"{split}.pt"


def pca_feature_path(config, split, dim, fold_name="split"):
    return feature_root(config, fold_name) / "pca" / f"dim_{dim}" / f"{split}.pt"


def pca_model_path(config, dim, fold_name="split"):
    return feature_root(config, fold_name) / "pca" / f"dim_{dim}" / "pca.pkl"


def save_feature_file(path, features, labels, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features.detach().cpu().float(),
            "labels": labels.detach().cpu().long(),
            "metadata": metadata or {},
        },
        path,
    )


def load_feature_file(path):
    data = torch.load(path, map_location="cpu")
    return data["features"].float(), data["labels"].long(), data.get("metadata", {})


def extract_features(backbone, loader, device):
    backbone.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_features = backbone(images).detach().cpu()
            features.append(batch_features)
            labels.append(batch_labels.detach().cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def save_raw_features(config, backbone, loaders, device, fold_name="split"):
    saved = {}
    for split, loader in loaders.items():
        features, labels = extract_features(backbone, loader, device)
        path = raw_feature_path(config, split, fold_name)
        metadata = {
            "split": split,
            "fold": fold_name,
            "feature_dim": features.size(1),
            "reducer": "none",
        }
        save_feature_file(path, features, labels, metadata)
        saved[split] = path
    return saved


def fit_and_save_pca_versions(config, dims, fold_name="split"):
    train_features, train_labels, _ = load_feature_file(raw_feature_path(config, "train", fold_name))
    val_features, val_labels, _ = load_feature_file(raw_feature_path(config, "val", fold_name))
    test_features, test_labels, _ = load_feature_file(raw_feature_path(config, "test", fold_name))

    feature_dim = train_features.size(1)
    max_components = min(train_features.size(0), feature_dim)

    for requested_dim in dims:
        dim = min(requested_dim, max_components)
        pca = PCA(n_components=dim, random_state=config.get("seed", 42))
        pca.fit(train_features.numpy())
        pca_path = pca_model_path(config, dim, fold_name)
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, pca_path)

        for split, features, labels in [
            ("train", train_features, train_labels),
            ("val", val_features, val_labels),
            ("test", test_features, test_labels),
        ]:
            reduced = torch.from_numpy(pca.transform(features.numpy()).astype(np.float32))
            metadata = {
                "split": split,
                "fold": fold_name,
                "feature_dim": feature_dim,
                "requested_dim": requested_dim,
                "reduced_dim": dim,
                "reducer": "pca",
            }
            save_feature_file(pca_feature_path(config, split, dim, fold_name), reduced, labels, metadata)


def features_exist(config, split_names=("train", "val", "test"), fold_name="split"):
    return all(raw_feature_path(config, split, fold_name).exists() for split in split_names)


def pca_features_exist(config, dim, split_names=("train", "val", "test"), fold_name="split"):
    return all(pca_feature_path(config, split, dim, fold_name).exists() for split in split_names)


def make_feature_loader(features, labels, batch_size, shuffle):
    dataset = TensorDataset(features.float(), labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_feature_loaders(config, feature_source, dim=None, fold_name="split", batch_size=None):
    feature_config = config.get("features", {})
    if batch_size is None:
        batch_size = feature_config.get("batch_size", config["dataset"].get("batch_size", 64))
    loaders = {}

    for split in ("train", "val", "test"):
        if feature_source == "raw":
            path = raw_feature_path(config, split, fold_name)
        elif feature_source == "pca":
            if dim is None:
                raise ValueError("dim is required when feature_source='pca'.")
            path = pca_feature_path(config, split, dim, fold_name)
        else:
            raise ValueError(f"Unsupported feature_source: {feature_source}")

        features, labels, _ = load_feature_file(path)
        loaders[f"{split}_loader"] = make_feature_loader(
            features, labels, batch_size, shuffle=(split == "train")
        )

    return loaders
