import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch

from main import run_single_split
from src.features import quantum_input_dim
from src.utils import load_config, save_json, set_seed


def _device_from_config(config):
    device_name = config["training"].get("device", "auto")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _tiny_hidden_dim(input_dim, num_classes, target_params):
    hidden_dim = round((target_params - num_classes) / (input_dim + num_classes + 1))
    return max(1, hidden_dim)


def _build_experiments(config):
    feature_config = config.get("features", {})
    model_config = config.get("model", {})
    tiny_config = model_config.get("mlp_tiny", {})
    qubits = feature_config.get("qubits", [4, 6, 8])
    encodings = feature_config.get("encodings", ["angle", "amplitude"])
    num_classes = model_config.get("num_classes", 10)
    tiny_target_params = tiny_config.get("target_params", 150)
    pca_dims = sorted(
        {quantum_input_dim(encoding, num_qubits) for encoding in encodings for num_qubits in qubits}
    )

    experiments = [
        {
            "name": "mlp_raw",
            "classifier": "mlp",
            "model_family": "mlp",
            "feature_source": "raw",
            "encoding": None,
            "num_qubits": None,
            "pca_dim": None,
<<<<<<< Updated upstream
            "hidden_layers": None,
            "dropout": None,
=======
>>>>>>> Stashed changes
        }
    ]

    for pca_dim in pca_dims:
        experiments.append(
            {
                "name": f"mlp_pca_dim_{pca_dim}",
                "classifier": "mlp",
                "model_family": "mlp",
                "feature_source": "pca",
                "encoding": None,
                "num_qubits": None,
                "pca_dim": pca_dim,
                "hidden_layers": None,
                "dropout": None,
            }
        )
        experiments.append(
            {
                "name": f"mlp_tiny_pca_dim_{pca_dim}",
                "classifier": "mlp",
                "model_family": "mlp_tiny",
                "feature_source": "pca",
                "encoding": None,
                "num_qubits": None,
                "pca_dim": pca_dim,
                "hidden_layers": [
                    _tiny_hidden_dim(pca_dim, num_classes, tiny_target_params)
                ],
                "dropout": tiny_config.get("dropout", 0.0),
            }
        )

    for encoding in encodings:
        for num_qubits in qubits:
            pca_dim = quantum_input_dim(encoding, num_qubits)
<<<<<<< Updated upstream
=======
            experiments.append(
                {
                    "name": f"mlp_pca_{encoding}_{num_qubits}q",
                    "classifier": "mlp",
                    "feature_source": "pca",
                    "encoding": encoding,
                    "num_qubits": num_qubits,
                    "pca_dim": pca_dim,
                }
            )
>>>>>>> Stashed changes
            experiments.append(
                {
                    "name": f"quantum_{encoding}_{num_qubits}q",
                    "classifier": "quantum",
                    "model_family": "quantum",
                    "feature_source": "pca",
                    "encoding": encoding,
                    "num_qubits": num_qubits,
                    "pca_dim": pca_dim,
<<<<<<< Updated upstream
                    "hidden_layers": None,
                    "dropout": None,
=======
>>>>>>> Stashed changes
                }
            )

    return experiments


def _config_for_experiment(base_config, experiment, output_root):
    config = copy.deepcopy(base_config)
    config["dataset"]["k_folds"] = 1
    config["training"]["output_dir"] = str(output_root / experiment["name"])
    config["model"]["classifier"] = experiment["classifier"]

    if experiment["classifier"] == "mlp":
        config["model"].setdefault("mlp", {})
        config["model"]["mlp"]["feature_source"] = experiment["feature_source"]
<<<<<<< Updated upstream
        if experiment.get("hidden_layers") is not None:
            config["model"]["mlp"]["hidden_layers"] = experiment["hidden_layers"]
        if experiment.get("dropout") is not None:
            config["model"]["mlp"]["dropout"] = experiment["dropout"]
=======
>>>>>>> Stashed changes
        if experiment["feature_source"] == "pca":
            config["model"]["mlp"]["reduced_dim"] = experiment["pca_dim"]
    else:
        config["model"].setdefault("quantum", {})
        config["model"]["quantum"]["encoding"] = experiment["encoding"]
        config["model"]["quantum"]["num_qubits"] = experiment["num_qubits"]

    return config


def _read_metrics(output_dir):
    metrics_path = Path(output_dir) / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_metrics(output_dir, metrics):
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def _add_duration(metrics, duration_seconds):
    if metrics is None:
        return None

    metrics["duration"] = {
        "seconds": duration_seconds,
        "minutes": duration_seconds / 60,
    }
    return metrics


def _summary_row(experiment, metrics):
    test = metrics.get("test", {}) if metrics else {}
    features = metrics.get("features", {}) if metrics else {}
    duration = metrics.get("duration", {}) if metrics else {}
    return {
        "experiment": experiment["name"],
        "classifier": experiment.get("model_family", experiment["classifier"]),
        "encoding": experiment.get("encoding") or "",
        "num_qubits": experiment.get("num_qubits") or "",
        "pca_dim": experiment.get("pca_dim") or "",
        "feature_source": features.get("feature_source", experiment.get("feature_source", "")),
        "feature_dim": features.get("feature_dim", ""),
        "trainable_params": metrics.get("trainable_params", "") if metrics else "",
        "test_loss": test.get("loss", ""),
        "test_accuracy": test.get("accuracy", ""),
        "duration_seconds": duration.get("seconds", ""),
        "duration_minutes": duration.get("minutes", ""),
    }


def _save_summary(rows, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(rows, output_root / "experiment_summary.json")

    csv_path = output_root / "experiment_summary.csv"
    fieldnames = [
        "experiment",
        "classifier",
        "encoding",
        "num_qubits",
        "pca_dim",
        "feature_source",
        "feature_dim",
        "trainable_params",
        "test_loss",
        "test_accuracy",
        "duration_seconds",
        "duration_minutes",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run all classifier experiments")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory where experiment folders will be created. Defaults to training.output_dir/experiments.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose metrics.json already exists.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    set_seed(base_config.get("seed", 42))
    device = _device_from_config(base_config)
    output_root = Path(args.output_root or Path(base_config["training"]["output_dir"]) / "experiments")
    experiments = _build_experiments(base_config)
    summary = []

    print(f"Using device: {device}")
    print(f"Running {len(experiments)} experiments")
    print(f"Output root: {output_root}")

    for index, experiment in enumerate(experiments, start=1):
        config = _config_for_experiment(base_config, experiment, output_root)
        output_dir = Path(config["training"]["output_dir"])
        metrics_path = output_dir / "metrics.json"

        experiment_device = torch.device("cpu") if experiment["classifier"] == "quantum" else device

        print(f"\n[{index}/{len(experiments)}] {experiment['name']}")
        print(f"Experiment device: {experiment_device}")
        if args.skip_existing and metrics_path.exists():
            print("Skipping existing experiment.")
        else:
            set_seed(config.get("seed", 42))
            start_time = time.perf_counter()
            run_single_split(config, experiment_device)
            duration_seconds = time.perf_counter() - start_time

            metrics = _read_metrics(output_dir)
            metrics = _add_duration(metrics, duration_seconds)
            if metrics is not None:
                _write_metrics(output_dir, metrics)
            print(
                f"Finished {experiment['name']} in "
                f"{duration_seconds / 60:.2f} minutes ({duration_seconds:.2f} seconds)."
            )

        metrics = _read_metrics(output_dir)
        summary.append(_summary_row(experiment, metrics))
        _save_summary(summary, output_root)

    print(f"\nSummary saved to {output_root / 'experiment_summary.json'}")
    print(f"Summary CSV saved to {output_root / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()
