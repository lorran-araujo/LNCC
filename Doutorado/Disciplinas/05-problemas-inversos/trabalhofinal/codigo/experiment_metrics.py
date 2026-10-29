"""Method-independent evaluation and result persistence for calibration experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosenzweig_dataset import PARAMETER_NAMES, ScenarioDataset, solve_model


METRIC_FIELDNAMES = [
    "run_id",
    "experiment_key",
    "method",
    "scenario_id",
    "case_id",
    "noise_replicate",
    "noise_fraction",
    "n_observations",
    "optimizer_seed",
    "status",
    "elapsed_seconds",
    "n_objective_evaluations",
    "weighted_observation_mse",
    "observation_rmse",
    "clean_trajectory_rmse",
    "relative_parameter_error",
    *[f"{name}_true" for name in PARAMETER_NAMES],
    *[f"{name}_estimated" for name in PARAMETER_NAMES],
    "method_config_json",
]


def evaluate_estimate(
    dataset: ScenarioDataset, scenario_id: int, estimated_parameters: np.ndarray
) -> dict[str, float]:
    """Evaluate estimates using common metrics available to all calibration methods."""
    true_parameters = dataset.parameters[scenario_id]
    initial_conditions = dataset.initial_conditions[scenario_id]
    observation_prediction = solve_model(estimated_parameters, initial_conditions, dataset.observation_time)
    clean_prediction = solve_model(estimated_parameters, initial_conditions, dataset.time)
    if observation_prediction is None or clean_prediction is None:
        return {
            "weighted_observation_mse": float("inf"),
            "observation_rmse": float("inf"),
            "clean_trajectory_rmse": float("inf"),
            "relative_parameter_error": float("inf"),
        }

    residual = observation_prediction - dataset.observations[scenario_id]
    standardized_residual = residual / np.maximum(dataset.noise_std[scenario_id, :, None], 1e-8)
    return {
        "weighted_observation_mse": float(np.mean(standardized_residual**2)),
        "observation_rmse": float(np.sqrt(np.mean(residual**2))),
        "clean_trajectory_rmse": float(
            np.sqrt(np.mean((clean_prediction - dataset.clean_states[scenario_id]) ** 2))
        ),
        "relative_parameter_error": float(
            np.linalg.norm(estimated_parameters - true_parameters) / np.linalg.norm(true_parameters)
        ),
    }


def make_metrics_row(
    *,
    run_id: str,
    experiment_key: str,
    method: str,
    scenario_id: int,
    optimizer_seed: int,
    status: str,
    elapsed_seconds: float,
    n_objective_evaluations: int,
    dataset: ScenarioDataset,
    estimated_parameters: np.ndarray,
    method_config: dict[str, Any],
) -> dict[str, Any]:
    """Build one schema-stable metrics row for any calibration approach."""
    metrics = evaluate_estimate(dataset, scenario_id, estimated_parameters)
    true_parameters = dataset.parameters[scenario_id]
    row: dict[str, Any] = {
        "run_id": run_id,
        "experiment_key": experiment_key,
        "method": method,
        "scenario_id": scenario_id,
        "case_id": int(dataset.case_ids[scenario_id]),
        "noise_replicate": int(dataset.noise_replicates[scenario_id]),
        "noise_fraction": float(dataset.noise_fractions[scenario_id]),
        "n_observations": int(dataset.n_observations[scenario_id]),
        "optimizer_seed": optimizer_seed,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "n_objective_evaluations": n_objective_evaluations,
        **metrics,
        "method_config_json": json.dumps(method_config, sort_keys=True),
    }
    row.update(dict(zip((f"{name}_true" for name in PARAMETER_NAMES), true_parameters, strict=True)))
    row.update(dict(zip((f"{name}_estimated" for name in PARAMETER_NAMES), estimated_parameters, strict=True)))
    return row


def make_experiment_key(
    *,
    method: str,
    case_id: int,
    noise_fraction: float,
    noise_replicate: int,
    optimizer_seed: int,
    method_config: dict[str, Any],
) -> str:
    """Create a stable identifier for one fully specified calibration run."""
    canonical_config = json.dumps(method_config, sort_keys=True, separators=(",", ":"))
    config_digest = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()[:12]
    return (
        f"{method}_c{case_id}_n{noise_fraction:.12g}_r{noise_replicate}"
        f"_s{optimizer_seed}_{config_digest}"
    )


def append_metrics(row: dict[str, Any], output_path: str | Path) -> Path:
    """Append a run to the shared metrics file, creating its stable header once."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_metrics_file(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in METRIC_FIELDNAMES})
    return path


def _migrate_metrics_file(path: Path) -> None:
    """Preserve prior results when the shared metrics schema gains new columns."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames == METRIC_FIELDNAMES:
            return
        existing_rows = list(reader)

    for row in existing_rows:
        if "optimizer_seed" not in row and "seed" in row:
            row["optimizer_seed"] = row["seed"]
        if not row.get("experiment_key"):
            row["experiment_key"] = _key_from_existing_row(row)
        for fieldname in METRIC_FIELDNAMES:
            row.setdefault(fieldname, "")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows({fieldname: row[fieldname] for fieldname in METRIC_FIELDNAMES} for row in existing_rows)


def migrate_metrics_file(path: str | Path) -> Path:
    """Upgrade a metrics file in place to the current result schema."""
    resolved_path = Path(path)
    _migrate_metrics_file(resolved_path)
    return resolved_path


def _key_from_existing_row(row: dict[str, str]) -> str:
    """Recover an experiment key for results written before deterministic IDs existed."""
    required_fields = ("method", "case_id", "noise_fraction", "noise_replicate", "optimizer_seed")
    if any(not row.get(field) for field in required_fields):
        return f"legacy_{row.get('run_id', 'unknown')}"
    try:
        method_config = json.loads(row.get("method_config_json", "{}"))
    except json.JSONDecodeError:
        method_config = {"legacy_run_id": row.get("run_id", "unknown")}
    return make_experiment_key(
        method=row["method"],
        case_id=int(row["case_id"]),
        noise_fraction=float(row["noise_fraction"]),
        noise_replicate=int(row["noise_replicate"]),
        optimizer_seed=int(row["optimizer_seed"]),
        method_config=method_config,
    )


def save_history(history: list[dict[str, float | int]], output_path: str | Path) -> Path:
    """Persist the optimization history for diagnostic plots and convergence analysis."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return path
