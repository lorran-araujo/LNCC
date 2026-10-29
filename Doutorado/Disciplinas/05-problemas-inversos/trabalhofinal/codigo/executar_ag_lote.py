"""Command-line entry point for resumable genetic-algorithm batches."""

from __future__ import annotations

import argparse
from pathlib import Path

from genetic_batch import BatchConfig, run_batch
from genetic_calibration import GeneticAlgorithmConfig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable genetic calibration batch.")
    parser.add_argument("--case-start", type=int, required=True, help="First physical case ID (inclusive).")
    parser.add_argument("--case-stop", type=int, required=True, help="Last physical case ID (exclusive).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--noise-fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--noise-replicates", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--optimizer-seeds", type=int, nargs="+", default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--generations", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.case_stop <= arguments.case_start:
        raise ValueError("case-stop must be greater than case-start.")

    project_dir = Path(__file__).resolve().parent.parent
    batch_config = BatchConfig(
        case_ids=tuple(range(arguments.case_start, arguments.case_stop)),
        noise_fractions=tuple(arguments.noise_fractions),
        noise_replicates=tuple(arguments.noise_replicates),
        optimizer_seeds=tuple(arguments.optimizer_seeds),
        workers=arguments.workers,
    )
    algorithm_config = GeneticAlgorithmConfig(
        population_size=arguments.population_size,
        generations=arguments.generations,
    )
    summary = run_batch(
        project_dir / "dados" / "piloto_trajectories.npz",
        project_dir / "dados" / "resultados",
        batch_config,
        algorithm_config,
    )
    print(
        f"requested={summary.requested} completed={summary.completed} skipped={summary.skipped}\n"
        f"batch={summary.batch_path}\nmetrics={summary.metrics_path}"
    )


if __name__ == "__main__":
    main()
