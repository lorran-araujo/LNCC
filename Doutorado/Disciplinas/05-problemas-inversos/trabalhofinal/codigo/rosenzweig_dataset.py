"""Reusable synthetic-data generator for the Rosenzweig-MacArthur model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc


PARAMETER_NAMES = ("r", "K", "a", "h", "e", "m")


@dataclass(frozen=True)
class ParameterBounds:
    """Pilot bounds for parameters with positive physical values."""

    lower: tuple[float, ...] = (0.25, 30.0, 0.005, 0.02, 0.10, 0.05)
    upper: tuple[float, ...] = (2.00, 300.0, 0.200, 1.00, 0.90, 1.00)


@dataclass(frozen=True)
class GenerationConfig:
    n_candidates: int = 256
    t_final: float = 30.0
    n_time: int = 1_000
    n_observations: int = 20
    noise_fractions: tuple[float, ...] = (0.01, 0.05, 0.10)
    n_noise_replicates: int = 5
    seed: int = 2026
    bounds: ParameterBounds = ParameterBounds()


@dataclass
class ScenarioDataset:
    time: np.ndarray
    observation_time: np.ndarray
    clean_states: np.ndarray
    observations: np.ndarray
    parameters: np.ndarray
    initial_conditions: np.ndarray
    noise_std: np.ndarray
    seeds: np.ndarray
    case_ids: np.ndarray
    noise_replicates: np.ndarray
    noise_fractions: np.ndarray
    n_observations: np.ndarray


def rosenzweig_macarthur(
    _time: float, state: np.ndarray, r: float, K: float, a: float, h: float, e: float, m: float
) -> list[float]:
    """Evaluate the prey-predator system right-hand side."""
    prey, predator = state
    predation = a * prey * predator / (1.0 + a * h * prey)
    return [r * prey * (1.0 - prey / K) - predation, e * predation - m * predator]


def sample_candidates(config: GenerationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sample parameters and initial conditions from a scrambled Sobol design."""
    design = qmc.Sobol(d=8, scramble=True, seed=config.seed).random(config.n_candidates)
    lower = np.asarray(config.bounds.lower)
    upper = np.asarray(config.bounds.upper)

    parameters = np.empty((config.n_candidates, len(PARAMETER_NAMES)))

    # Positive rates span orders of magnitude, so sample them in logarithmic scale.
    log_indices = np.array([0, 1, 2, 3, 5])
    parameters[:, log_indices] = np.exp(
        np.log(lower[log_indices])
        + design[:, log_indices] * (np.log(upper[log_indices]) - np.log(lower[log_indices]))
    )
    parameters[:, 4] = lower[4] + design[:, 4] * (upper[4] - lower[4])

    carrying_capacity = parameters[:, 1]
    initial_conditions = np.column_stack(
        (
            carrying_capacity * (0.05 + 0.45 * design[:, 6]),
            carrying_capacity * (0.01 + 0.19 * design[:, 7]),
        )
    )
    return parameters, initial_conditions


def has_feasible_coexistence(parameters: np.ndarray) -> bool:
    """Keep pilot scenarios with an internal prey-predator equilibrium."""
    _r, carrying_capacity, attack_rate, handling_time, efficiency, mortality = parameters
    if efficiency <= mortality * handling_time:
        return False

    prey_equilibrium = mortality / (attack_rate * (efficiency - mortality * handling_time))
    return 0.05 * carrying_capacity < prey_equilibrium < 0.95 * carrying_capacity


def simulate_scenario(
    parameters: np.ndarray, initial_conditions: np.ndarray, config: GenerationConfig
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve one candidate trajectory, returning None for numerically invalid runs."""
    time = np.linspace(0.0, config.t_final, config.n_time)
    states = solve_model(parameters, initial_conditions, time)
    if states is None:
        return None

    carrying_capacity = parameters[1]
    if np.any(states < 0.0) or np.max(states) > 10.0 * carrying_capacity:
        return None
    return time, states


def solve_model(
    parameters: np.ndarray, initial_conditions: np.ndarray, evaluation_time: np.ndarray
) -> np.ndarray | None:
    """Solve the model at specified times, returning None after an integration failure."""
    solution = solve_ivp(
        rosenzweig_macarthur,
        (evaluation_time[0], evaluation_time[-1]),
        initial_conditions,
        t_eval=evaluation_time,
        args=tuple(parameters),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        return None

    states = solution.y
    if not np.all(np.isfinite(states)):
        return None
    return states


def generate_pilot_dataset(config: GenerationConfig = GenerationConfig()) -> ScenarioDataset:
    """Generate cases, observation replicas, and noise levels for inverse experiments."""
    if not config.noise_fractions or any(noise_fraction < 0.0 for noise_fraction in config.noise_fractions):
        raise ValueError("noise_fractions must contain non-negative values.")
    if config.n_noise_replicates < 1:
        raise ValueError("n_noise_replicates must be at least one.")
    candidates, initial_conditions = sample_candidates(config)
    observation_indices = np.linspace(0, config.n_time - 1, config.n_observations, dtype=int)
    rng = np.random.default_rng(config.seed)

    clean_states: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    parameters: list[np.ndarray] = []
    initials: list[np.ndarray] = []
    noise_stds: list[np.ndarray] = []
    seeds: list[int] = []
    case_ids: list[int] = []
    noise_replicates: list[int] = []
    noise_fractions: list[float] = []
    observation_counts: list[int] = []
    time: np.ndarray | None = None

    next_case_id = 0
    for candidate, initial in zip(candidates, initial_conditions, strict=True):
        if not has_feasible_coexistence(candidate):
            continue
        simulated = simulate_scenario(candidate, initial, config)
        if simulated is None:
            continue

        time, states = simulated
        # One physical case yields several independently noisy observation sets.
        state_range = np.ptp(states, axis=1)
        for noise_fraction in config.noise_fractions:
            noise_std = noise_fraction * np.maximum(state_range, 1e-8)
            for noise_replicate in range(config.n_noise_replicates):
                observation_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
                observation_rng = np.random.default_rng(observation_seed)
                noisy_observations = states[:, observation_indices] + observation_rng.normal(
                    loc=0.0, scale=noise_std[:, None], size=(2, config.n_observations)
                )
                # Population measurements cannot be negative; retain the noise scale in metadata.
                noisy_observations = np.maximum(noisy_observations, 0.0)

                clean_states.append(states)
                observations.append(noisy_observations)
                parameters.append(candidate)
                initials.append(initial)
                noise_stds.append(noise_std)
                seeds.append(observation_seed)
                case_ids.append(next_case_id)
                noise_replicates.append(noise_replicate)
                noise_fractions.append(noise_fraction)
                observation_counts.append(config.n_observations)
        next_case_id += 1

    if not clean_states or time is None:
        raise RuntimeError("No viable scenario was generated. Widen the pilot bounds or revise the filters.")

    return ScenarioDataset(
        time=time,
        observation_time=time[observation_indices],
        clean_states=np.asarray(clean_states),
        observations=np.asarray(observations),
        parameters=np.asarray(parameters),
        initial_conditions=np.asarray(initials),
        noise_std=np.asarray(noise_stds),
        seeds=np.asarray(seeds, dtype=np.uint32),
        case_ids=np.asarray(case_ids, dtype=int),
        noise_replicates=np.asarray(noise_replicates, dtype=int),
        noise_fractions=np.asarray(noise_fractions, dtype=float),
        n_observations=np.asarray(observation_counts, dtype=int),
    )


def save_dataset(dataset: ScenarioDataset, output_dir: str | Path) -> tuple[Path, Path]:
    """Save numerical arrays as NPZ and one inspectable metadata row per scenario."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    arrays_path = output_path / "piloto_trajectories.npz"
    metadata_path = output_path / "piloto_metadata.csv"

    np.savez_compressed(
        arrays_path,
        time=dataset.time,
        observation_time=dataset.observation_time,
        clean_states=dataset.clean_states,
        observations=dataset.observations,
        parameters=dataset.parameters,
        initial_conditions=dataset.initial_conditions,
        noise_std=dataset.noise_std,
        seeds=dataset.seeds,
        case_ids=dataset.case_ids,
        noise_replicates=dataset.noise_replicates,
        noise_fractions=dataset.noise_fractions,
        n_observations=dataset.n_observations,
    )

    fieldnames = [
        "scenario_id",
        "case_id",
        "noise_replicate",
        "noise_fraction",
        "n_observations",
        *PARAMETER_NAMES,
        "N0",
        "P0",
        "sigma_N",
        "sigma_P",
        "noise_seed",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_id, (theta, initial, noise, seed, case_id, noise_replicate, noise_fraction, n_observations) in enumerate(
            zip(
                dataset.parameters,
                dataset.initial_conditions,
                dataset.noise_std,
                dataset.seeds,
                dataset.case_ids,
                dataset.noise_replicates,
                dataset.noise_fractions,
                dataset.n_observations,
                strict=True,
            )
        ):
            writer.writerow(
                {
                    "scenario_id": scenario_id,
                    "case_id": int(case_id),
                    "noise_replicate": int(noise_replicate),
                    "noise_fraction": float(noise_fraction),
                    "n_observations": int(n_observations),
                    **dict(zip(PARAMETER_NAMES, theta, strict=True)),
                    "N0": initial[0],
                    "P0": initial[1],
                    "sigma_N": noise[0],
                    "sigma_P": noise[1],
                    "noise_seed": int(seed),
                }
            )
    return arrays_path, metadata_path


def load_dataset(arrays_path: str | Path) -> ScenarioDataset:
    """Load a dataset previously materialized by :func:`save_dataset`."""
    with np.load(arrays_path) as arrays:
        return ScenarioDataset(
            time=arrays["time"],
            observation_time=arrays["observation_time"],
            clean_states=arrays["clean_states"],
            observations=arrays["observations"],
            parameters=arrays["parameters"],
            initial_conditions=arrays["initial_conditions"],
            noise_std=arrays["noise_std"],
            seeds=arrays["seeds"],
            case_ids=arrays["case_ids"],
            noise_replicates=arrays["noise_replicates"],
            noise_fractions=arrays["noise_fractions"],
            n_observations=arrays["n_observations"],
        )


def plot_scenario(dataset: ScenarioDataset, scenario_id: int = 0) -> None:
    """Plot clean and noisy prey-predator data for one generated scenario."""
    if not 0 <= scenario_id < len(dataset.parameters):
        raise IndexError(f"scenario_id must be in [0, {len(dataset.parameters) - 1}].")

    prey, predator = dataset.clean_states[scenario_id]
    prey_observed, predator_observed = dataset.observations[scenario_id]
    cor_presas = "#2563EB"
    cor_predadores = "#E76F51"

    plt.figure(figsize=(11, 5.5))
    plt.plot(dataset.time, prey, color=cor_presas, linewidth=2.5, label="Presas - solucao numerica")
    plt.plot(dataset.time, predator, color=cor_predadores, linewidth=2.5, label="Predadores - solucao numerica")
    plt.scatter(
        dataset.observation_time,
        prey_observed,
        color=cor_presas,
        edgecolors="white",
        linewidths=0.9,
        s=65,
        marker="o",
        zorder=5,
        label="Presas - observacoes",
    )
    plt.scatter(
        dataset.observation_time,
        predator_observed,
        color=cor_predadores,
        edgecolors="white",
        linewidths=0.9,
        s=65,
        marker="s",
        zorder=5,
        label="Predadores - observacoes",
    )
    plt.title(f"Cenario {scenario_id}: dados sinteticos de Rosenzweig-MacArthur", fontsize=15, fontweight="bold", pad=16)
    plt.xlabel("Tempo", fontsize=12)
    plt.ylabel("Populacao", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
    plt.legend(fontsize=10, loc="best", frameon=True, fancybox=True)
    plt.tick_params(axis="both", labelsize=10)
    axis = plt.gca()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    plt.xlim(dataset.time[0], dataset.time[-1])
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()
