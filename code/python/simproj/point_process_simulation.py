"""Facade for conditional point-process perturbation experiments."""
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

    # Multi-exponential Hawkes parameters.
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

    process: str = "hawkes"

    # Affine counting-process parameters: lambda(N_t) = b + a * N_t.
    a: float = 0.05
    b: float = 1.0


def _process_kind(cfg: PointProcessSimulationConfig) -> str:
    kind = str(cfg.process).strip().lower()
    if kind in {"hawkes", "multi_exponential_hawkes", "multi-exponential-hawkes"}:
        return "hawkes"
    if kind in {"affine", "affine_counting", "affine-counting", "affine_count"}:
        return "affine"
    raise ValueError("process must be 'hawkes' or 'affine'")


def _make_hawkes(cfg: PointProcessSimulationConfig):
    if cfg.stationary:
        return _native.MultiExponentialHawkes.with_stationary_state(
            cfg.mu, list(cfg.alpha), list(cfg.beta),
        )
    return _native.MultiExponentialHawkes(cfg.mu, list(cfg.alpha), list(cfg.beta))


def _make_affine(cfg: PointProcessSimulationConfig):
    return _native.AffineCountingProcess(float(cfg.a), float(cfg.b))


def _make_process(cfg: PointProcessSimulationConfig, kind: str):
    if kind == "hawkes":
        return _make_hawkes(cfg)
    if kind == "affine":
        return _make_affine(cfg)
    raise ValueError("process must be 'hawkes' or 'affine'")


def _simulate_baseline(process, cfg: PointProcessSimulationConfig, kind: str):
    if kind == "hawkes":
        return _native.simulate_hawkes_result(process, cfg.time_horizon, cfg.seed)
    return _native.simulate_affine_counting_process(process, cfg.time_horizon, cfg.seed)


def _make_context(process, baseline_times: np.ndarray, perturbation, cfg, kind: str):
    ctx_cls = (
        _native.ConditionalHawkesSimulationContext
        if kind == "hawkes"
        else _native.ConditionalAffineCountingSimulationContext
    )
    return ctx_cls(
        process,
        [baseline_times.tolist()],
        cfg.time_horizon,
        new_externals=perturbation,
    )


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

    kind = _process_kind(cfg)
    process = _make_process(cfg, kind)
    baseline = _simulate_baseline(process, cfg, kind)
    baseline_times = np.asarray(baseline.times(), dtype=np.float64)

    perturbation = _make_perturbation_events(cfg)
    ctx = _make_context(process, baseline_times, perturbation, cfg, kind)

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
        "process_kind": np.array([kind]),
        "affine_a": np.array([cfg.a], dtype=np.float64),
        "affine_b": np.array([cfg.b], dtype=np.float64),
    }


def save(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
