"""Facade for conditional Hawkes point-process perturbation experiments."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from . import _native


@dataclass
class PointProcessSimulationConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500

    mu: float = 1.0
    alpha: list = field(default_factory=lambda: [0.065, 0.2, 0.325, 0.65])
    beta: list = field(default_factory=lambda: [0.15, 0.60, 2.5, 10.0])
    stationary: bool = True

    # Float for one added event; list/ndarray for an explicit perturbation block.
    perturbation_time: Union[float, list, np.ndarray] = 10.0

    # False gives independent Monte Carlo paths. True couples acceptance
    # randoms across paths, useful for monotone path comparisons.
    shared_acceptance: bool = False

    seed: int = 42


def _make_hawkes(cfg: PointProcessSimulationConfig):
    if cfg.stationary:
        return _native.MultiExponentialHawkes.with_stationary_state(
            cfg.mu, list(cfg.alpha), list(cfg.beta),
        )
    return _native.MultiExponentialHawkes(cfg.mu, list(cfg.alpha), list(cfg.beta))


def _perturbation_times(cfg: PointProcessSimulationConfig) -> np.ndarray:
    times = np.atleast_1d(np.asarray(cfg.perturbation_time, dtype=np.float64))
    if times.ndim != 1:
        raise ValueError("perturbation_time must be a scalar or one-dimensional array")
    if np.any(~np.isfinite(times)):
        raise ValueError("perturbation times must be finite")
    if np.any(times < 0.0) or np.any(times > cfg.time_horizon):
        raise ValueError("perturbation times must lie inside [0, time_horizon]")
    return np.sort(times)


def _make_perturbation_events(cfg: PointProcessSimulationConfig):
    return _native.create_meta_orders_from_times(
        _perturbation_times(cfg), target_dim=0, total_dims=1,
    )


def _pad_time_paths(paths: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(path) for path in paths], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) else 0
    padded = np.full((len(paths), max_len), np.nan, dtype=np.float64)
    for row, path in enumerate(paths):
        padded[row, : len(path)] = path
    return padded, lengths


def run(cfg: PointProcessSimulationConfig) -> dict:
    if cfg.n_simulations < 0:
        raise ValueError("n_simulations must be non-negative")
    if cfg.time_horizon <= 0.0:
        raise ValueError("time_horizon must be positive")

    hawkes = _make_hawkes(cfg)
    baseline = _native.simulate_hawkes_result(hawkes, cfg.time_horizon, cfg.seed)
    baseline_times = np.asarray(baseline.times(), dtype=np.float64)

    perturbation = _make_perturbation_events(cfg)
    ctx = _native.ConditionalHawkesSimulationContext(
        hawkes,
        [baseline_times.tolist()],
        cfg.time_horizon,
        new_externals=perturbation,
    )

    sim_seed = int(cfg.seed) + 1
    paths = [
        np.asarray(path, dtype=np.float64)
        for path in ctx.simulate_many_times(
            cfg.n_simulations,
            base_seed=sim_seed,
            shared_acceptance=cfg.shared_acceptance,
        )
    ]
    perturbed_times, perturbed_lengths = _pad_time_paths(paths)

    return {
        "baseline_times": baseline_times,
        "perturbation_times": np.asarray(perturbation.times(), dtype=np.float64),
        "perturbed_times": perturbed_times,
        "perturbed_lengths": perturbed_lengths,
        "baseline_count": np.array([len(baseline_times)], dtype=np.int64),
        "time_horizon": np.array([cfg.time_horizon], dtype=np.float64),
    }


def save(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
