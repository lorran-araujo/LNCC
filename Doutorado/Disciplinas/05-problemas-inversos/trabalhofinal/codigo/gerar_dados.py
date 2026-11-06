"""Command-line entry point for reproducible Rosenzweig-MacArthur datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from rosenzweig_dataset import GenerationConfig, generate_pilot_dataset, save_dataset


def parse_arguments() -> argparse.Namespace:
    """Parse the dataset definition from the command line."""
    parser = argparse.ArgumentParser(description="Generate a synthetic Rosenzweig-MacArthur dataset.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for the NPZ and metadata CSV.")
    parser.add_argument("--name", default="rosenzweig_piloto", help="Dataset name used as the file-name prefix.")
    parser.add_argument("--candidates", type=int, default=256, help="Number of Sobol candidates to simulate.")
    parser.add_argument("--observations", type=int, default=20, help="Number of observations per trajectory.")
    parser.add_argument("--noise-fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--noise-replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    """Generate and persist one self-contained dataset."""
    arguments = parse_arguments()
    project_dir = Path(__file__).resolve().parent.parent
    output_dir = arguments.output_dir or project_dir / "dados" / "datasets"
    config = GenerationConfig(
        n_candidates=arguments.candidates,
        n_observations=arguments.observations,
        noise_fractions=tuple(arguments.noise_fractions),
        n_noise_replicates=arguments.noise_replicates,
        seed=arguments.seed,
    )
    dataset = generate_pilot_dataset(config)
    arrays_path, metadata_path = save_dataset(dataset, output_dir, arguments.name)
    print(
        f"physical_cases={len(set(dataset.case_ids))} observation_scenarios={len(dataset.parameters)}\n"
        f"dataset={arrays_path}\nmetadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
