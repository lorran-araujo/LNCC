"""Resumable batched execution for genetic calibration experiments."""

from __future__ import annotations

import csv
import fcntl
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from experiment_metrics import (
    METRIC_FIELDNAMES,
    append_metrics,
    make_experiment_key,
    make_metrics_row,
    migrate_metrics_file,
    save_history,
)
from genetic_calibration import GeneticAlgorithmConfig, calibrate_genetic_algorithm
from rosenzweig_dataset import ScenarioDataset, load_dataset


@dataclass(frozen=True)
class BatchConfig:
    case_ids: tuple[int, ...]
    noise_fractions: tuple[float, ...]
    noise_replicates: tuple[int, ...]
    optimizer_seeds: tuple[int, ...]
    workers: int = 4


@dataclass(frozen=True)
class BatchSummary:
    requested: int
    skipped: int
    completed: int
    batch_path: Path
    metrics_path: Path


_DATASET: ScenarioDataset | None = None
_BASE_CONFIG: GeneticAlgorithmConfig | None = None


def run_batch(
    dataset_path: str | Path,
    results_dir: str | Path,
    batch_config: BatchConfig,
    base_config: GeneticAlgorithmConfig = GeneticAlgorithmConfig(),
) -> BatchSummary:
    """Run missing combinations and consolidate their metrics safely."""
    if batch_config.workers < 1:
        raise ValueError("workers must be at least one.")

    dataset_path = Path(dataset_path)
    results_path = Path(results_dir)
    metrics_path = results_path / "metricas.csv"
    lots_path = results_path / "lotes"
    history_path = results_path / "historicos"
    dataset = load_dataset(dataset_path)
    scenario_ids = _select_scenarios(dataset, batch_config)
    batch_path = lots_path / _batch_filename(batch_config.case_ids)

    completed_keys = _completed_keys([metrics_path, *lots_path.glob("*.csv")])
    tasks = []
    for scenario_id in scenario_ids:
        for optimizer_seed in batch_config.optimizer_seeds:
            run_config = replace(base_config, seed=optimizer_seed)
            method_config = asdict(run_config)
            key = make_experiment_key(
                method="genetic_algorithm",
                case_id=int(dataset.case_ids[scenario_id]),
                noise_fraction=float(dataset.noise_fractions[scenario_id]),
                noise_replicate=int(dataset.noise_replicates[scenario_id]),
                optimizer_seed=optimizer_seed,
                method_config=method_config,
            )
            if key not in completed_keys:
                tasks.append((scenario_id, optimizer_seed, key))

    if tasks:
        worker_results = _execute_tasks(dataset_path, base_config, tasks, batch_config.workers)
        for row, history in worker_results:
            append_metrics(row, batch_path)
            save_history(history, history_path / f"{row['run_id']}.csv")

    consolidate_batches(results_path)
    requested = len(scenario_ids) * len(batch_config.optimizer_seeds)
    return BatchSummary(
        requested=requested,
        skipped=requested - len(tasks),
        completed=len(tasks),
        batch_path=batch_path,
        metrics_path=metrics_path,
    )


def consolidate_batches(results_dir: str | Path) -> Path:
    """Merge batch files into one deduplicated metrics file under an exclusive lock."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    metrics_path = results_path / "metricas.csv"
    lots_path = results_path / "lotes"
    lock_path = results_path / ".metricas.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            source_paths = [metrics_path, *sorted(lots_path.glob("*.csv"))]
            rows_by_key: dict[str, dict[str, str]] = {}
            for source_path in source_paths:
                if not source_path.exists():
                    continue
                migrate_metrics_file(source_path)
                with source_path.open(newline="", encoding="utf-8") as file:
                    for row in csv.DictReader(file):
                        rows_by_key[row["experiment_key"]] = row

            temporary_path = metrics_path.with_suffix(".tmp")
            with temporary_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
                writer.writeheader()
                for key in sorted(rows_by_key):
                    writer.writerow({field: rows_by_key[key].get(field, "") for field in METRIC_FIELDNAMES})
            temporary_path.replace(metrics_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return metrics_path


def _select_scenarios(dataset: ScenarioDataset, batch_config: BatchConfig) -> list[int]:
    selected = [
        scenario_id
        for scenario_id in range(len(dataset.parameters))
        if dataset.case_ids[scenario_id] in batch_config.case_ids
        and np.isclose(dataset.noise_fractions[scenario_id], batch_config.noise_fractions).any()
        and dataset.noise_replicates[scenario_id] in batch_config.noise_replicates
    ]
    if not selected:
        raise ValueError("The batch filters did not select any observation scenario.")
    return selected


def _batch_filename(case_ids: Iterable[int]) -> str:
    sorted_case_ids = sorted(case_ids)
    if not sorted_case_ids:
        raise ValueError("case_ids cannot be empty.")
    return f"ag_cases_{sorted_case_ids[0]:03d}_{sorted_case_ids[-1]:03d}.csv"


def _completed_keys(paths: Iterable[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        migrate_metrics_file(path)
        with path.open(newline="", encoding="utf-8") as file:
            keys.update(row["experiment_key"] for row in csv.DictReader(file) if row.get("experiment_key"))
    return keys


def _execute_tasks(
    dataset_path: Path,
    base_config: GeneticAlgorithmConfig,
    tasks: list[tuple[int, int, str]],
    workers: int,
) -> list[tuple[dict[str, object], list[dict[str, float | int]]]]:
    if workers == 1:
        _initialize_worker(dataset_path, base_config)
        return [_run_task(task) for task in tasks]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(dataset_path, base_config),
    ) as executor:
        return list(executor.map(_run_task, tasks))


def _initialize_worker(dataset_path: Path, base_config: GeneticAlgorithmConfig) -> None:
    global _DATASET, _BASE_CONFIG
    _DATASET = load_dataset(dataset_path)
    _BASE_CONFIG = base_config


def _run_task(task: tuple[int, int, str]) -> tuple[dict[str, object], list[dict[str, float | int]]]:
    if _DATASET is None or _BASE_CONFIG is None:
        raise RuntimeError("Worker was not initialized.")
    scenario_id, optimizer_seed, experiment_key = task
    config = replace(_BASE_CONFIG, seed=optimizer_seed)
    started_at = perf_counter()
    result = calibrate_genetic_algorithm(_DATASET, scenario_id, config)
    row = make_metrics_row(
        run_id=experiment_key,
        experiment_key=experiment_key,
        method="genetic_algorithm",
        scenario_id=scenario_id,
        optimizer_seed=optimizer_seed,
        status="completed",
        elapsed_seconds=perf_counter() - started_at,
        n_objective_evaluations=result.n_objective_evaluations,
        dataset=_DATASET,
        estimated_parameters=result.parameters,
        method_config=asdict(config),
    )
    return row, result.history
