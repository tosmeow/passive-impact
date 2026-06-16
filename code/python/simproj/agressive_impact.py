"""Facade for the aggressive impact experiment (hybrid model)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np

from . import _native


@dataclass
class AggressiveImpactConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500
    initial_queue_size: int = 200
    counterfactual: bool = False     # False: with us | True: without us

    bar_kappa: float = 0.01         # constant weight for propagated metaorders

    # Hawkes
    mu: float = 1.0
    alpha: list = field(default_factory=lambda: [0.065, 0.2, 0.325, 0.65])
    beta: list = field(default_factory=lambda: [0.15, 0.60, 2.5, 10.0])

    # Affine queue
    a_l: float = 100.0
    b_l: float = -0.275
    a_c: float = 2.0
    b_c: float = 0.125

    # Kappa(q) — hybrid instantaneous queue-dependent correction
    kappa: Callable[[float], float] = field(
        default_factory=lambda: lambda q: -0.001 * q
    )

    # Metaorder (aggressive metaorders are dim=2 = market orders)
    metaorder: Union[int, list, np.ndarray] = 200
    metaorder_window: tuple = (1.0, 75.0)

    seed: int = 42


def _make_meta_orders(cfg: AggressiveImpactConfig):
    if isinstance(cfg.metaorder, int):
        meta = _native.create_meta_orders(cfg.metaorder, *cfg.metaorder_window)
        return _native.events_to_dim(meta, target_dim=2, total_dims=3)
    times = np.asarray(cfg.metaorder, dtype=np.float64)
    return _native.create_meta_orders_from_times(times, target_dim=2, total_dims=3)


def _run_single_direction(cfg: AggressiveImpactConfig, direction: str) -> dict:
    hawkes = _native.MultiExponentialHawkes.with_stationary_state(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
    )
    market = _native.simulate_hawkes_as_market_orders(hawkes, cfg.time_horizon, cfg.seed)
    process = _native.AffineQueueProcess.new_queue(
        float(cfg.initial_queue_size), cfg.a_l, cfg.b_l, cfg.a_c, cfg.b_c,
    )
    meta = _make_meta_orders(cfg)
    bar_q_external = _native.merge_events(meta, market)
    q_external = market

    market_times = list(market.times())
    meta_times = list(meta.times())

    eval_entries = sorted([(t, True) for t in market_times] + [(t, False) for t in meta_times])
    eval_times = np.array([t for t, _ in eval_entries], dtype=np.float64)
    is_market_order = [b for _, b in eval_entries]

    n_times = len(eval_times)
    queue_paths = np.empty((n_times, cfg.n_simulations + 1), dtype=np.uint32)
    impact_paths = np.empty((n_times, cfg.n_simulations), dtype=np.float64)

    if direction == "with":
        q_internal = _native.simulate_with_externals(
            process, cfg.time_horizon, q_external, cfg.seed,
        )
        q_full = _native.merge_events(q_internal, q_external)
        cond_by_dim = [
            list(arr) for arr in _native.extract_events_by_dim(q_internal, 3, 2)
        ]
        reference_samples = _native.sample_queue_at_times(
            q_full, cfg.initial_queue_size, eval_times,
        )
        queue_paths[:, 0] = reference_samples
        cond_externals = q_external
        new_externals = bar_q_external
        simulating_bar_q = True
    elif direction == "without":
        bar_q_internal = _native.simulate_with_externals(
            process, cfg.time_horizon, bar_q_external, cfg.seed,
        )
        bar_q_full = _native.merge_events(bar_q_internal, bar_q_external)
        cond_by_dim = [
            list(arr) for arr in _native.extract_events_by_dim(bar_q_internal, 3, 2)
        ]
        reference_samples = _native.sample_queue_at_times(
            bar_q_full, cfg.initial_queue_size, eval_times,
        )
        queue_paths[:, 0] = reference_samples
        cond_externals = bar_q_external
        new_externals = q_external
        simulating_bar_q = False
    else:
        raise ValueError(f"direction must be 'with' or 'without'; got {direction!r}")

    ctx = _native.ConditionalSimulationContext(
        process, cond_by_dim,
        cfg.time_horizon,
        cond_externals=cond_externals,
        new_externals=new_externals,
    )
    for sim_idx in range(cfg.n_simulations):
        sim_samples = ctx.simulate_queue_at_times(
            eval_times, cfg.initial_queue_size, seed=sim_idx,
        )
        queue_paths[:, sim_idx + 1] = sim_samples

        if simulating_bar_q:
            q_samples = reference_samples
            q_bar_samples = sim_samples
        else:
            q_samples = sim_samples
            q_bar_samples = reference_samples

        result = _native.aggressive_impact_from_queue_samples(
            q_samples=q_samples, q_bar_samples=q_bar_samples,
            eval_times=eval_times, is_market_order=is_market_order,
            hawkes=hawkes, kappa=cfg.kappa,
            bar_kappa=cfg.bar_kappa,
        )
        impact_paths[:, sim_idx] = result.impact()

    return {
        "times": eval_times,
        "queue_paths": queue_paths,
        "impact_paths": impact_paths,
        "event_types": np.array([1.0 if b else 0.0 for b in is_market_order]),
        "bar_kappa": np.array([cfg.bar_kappa], dtype=np.float64),
    }


def run(cfg: AggressiveImpactConfig) -> dict:
    return _run_single_direction(cfg, "without" if cfg.counterfactual else "with")


def save(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if "with" in result and "without" in result:
        save(result["with"], os.path.join(output_dir, "with"))
        save(result["without"], os.path.join(output_dir, "without"))
        return
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
