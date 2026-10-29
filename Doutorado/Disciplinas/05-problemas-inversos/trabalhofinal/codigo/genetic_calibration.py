"""Genetic calibration for one Rosenzweig-MacArthur observation scenario."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rosenzweig_dataset import ParameterBounds, ScenarioDataset, solve_model


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = 80
    generations: int = 150
    elite_count: int = 2
    tournament_size: int = 3
    crossover_probability: float = 0.90
    mutation_probability: float = 0.15
    mutation_scale: float = 0.08
    seed: int = 2027
    bounds: ParameterBounds = ParameterBounds()


@dataclass
class CalibrationResult:
    parameters: np.ndarray
    objective_value: float
    n_objective_evaluations: int
    history: list[dict[str, float | int]]


def decode_parameters(encoded: np.ndarray, bounds: ParameterBounds) -> np.ndarray:
    """Map unit-hypercube chromosomes to physically meaningful parameters."""
    lower = np.asarray(bounds.lower)
    upper = np.asarray(bounds.upper)
    encoded = np.asarray(encoded)
    parameters = np.empty_like(encoded, dtype=float)
    log_indices = np.array([0, 1, 2, 3, 5])
    parameters[..., log_indices] = np.exp(
        np.log(lower[log_indices])
        + encoded[..., log_indices] * (np.log(upper[log_indices]) - np.log(lower[log_indices]))
    )
    parameters[..., 4] = lower[4] + encoded[..., 4] * (upper[..., 4] - lower[..., 4])
    return parameters


def calibrate_genetic_algorithm(
    dataset: ScenarioDataset, scenario_id: int, config: GeneticAlgorithmConfig = GeneticAlgorithmConfig()
) -> CalibrationResult:
    """Estimate one scenario's parameters using a bounded real-valued genetic algorithm."""
    if config.elite_count < 1 or config.elite_count >= config.population_size:
        raise ValueError("elite_count must be at least one and smaller than population_size.")
    if not 0 <= scenario_id < len(dataset.parameters):
        raise IndexError(f"scenario_id must be in [0, {len(dataset.parameters) - 1}].")

    observed = dataset.observations[scenario_id]
    initial_conditions = dataset.initial_conditions[scenario_id]
    noise_std = np.maximum(dataset.noise_std[scenario_id], 1e-8)
    rng = np.random.default_rng(config.seed)
    evaluations = 0

    def objective(chromosome: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        parameters = decode_parameters(np.clip(chromosome, 0.0, 1.0), config.bounds)
        prediction = solve_model(parameters, initial_conditions, dataset.observation_time)
        if prediction is None or np.any(prediction < 0.0):
            return 1e12
        residual = (prediction - observed) / noise_std[:, None]
        return float(np.mean(residual**2))

    def evaluate(population: np.ndarray) -> np.ndarray:
        return np.asarray([objective(chromosome) for chromosome in population])

    def tournament(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        candidates = rng.integers(0, len(population), size=config.tournament_size)
        return population[candidates[np.argmin(fitness[candidates])]]

    population = rng.random((config.population_size, 6))
    fitness = evaluate(population)
    history: list[dict[str, float | int]] = []

    for generation in range(config.generations + 1):
        order = np.argsort(fitness)
        history.append(
            {
                "generation": generation,
                "best_objective": float(fitness[order[0]]),
                "mean_objective": float(np.mean(fitness)),
                "std_objective": float(np.std(fitness)),
                "n_objective_evaluations": evaluations,
            }
        )
        if generation == config.generations:
            break

        next_population = [population[index].copy() for index in order[: config.elite_count]]
        while len(next_population) < config.population_size:
            parent_a = tournament(population, fitness)
            parent_b = tournament(population, fitness)
            child_a, child_b = parent_a.copy(), parent_b.copy()
            if rng.random() < config.crossover_probability:
                blend = rng.random(6)
                child_a = blend * parent_a + (1.0 - blend) * parent_b
                child_b = blend * parent_b + (1.0 - blend) * parent_a

            for child in (child_a, child_b):
                mutation_mask = rng.random(6) < config.mutation_probability
                child[mutation_mask] += rng.normal(0.0, config.mutation_scale, size=mutation_mask.sum())
                next_population.append(np.clip(child, 0.0, 1.0))
                if len(next_population) == config.population_size:
                    break

        population = np.asarray(next_population)
        fitness = evaluate(population)

    best_index = int(np.argmin(fitness))
    return CalibrationResult(
        parameters=decode_parameters(population[best_index], config.bounds),
        objective_value=float(fitness[best_index]),
        n_objective_evaluations=evaluations,
        history=history,
    )
