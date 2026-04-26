"""Facade for the aggressive impact experiment (propagator + hybrid models)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from . import _native


@dataclass
class AggressiveImpactConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500
    initial_queue_size: int = 200

    model: str = "propagator"       # "propagator" | "hybrid"
    bar_kappa: Optional[float] = None  # required when model == "hybrid"

    # Hawkes
    mu: float = 1.0
    alpha: list = field(default_factory=lambda: [0.065, 0.2, 0.325, 0.65])
    beta: list = field(default_factory=lambda: [0.15, 0.60, 2.5, 10.0])

    # Affine queue
    a_l: float = 100.0
    b_l: float = -0.275
    a_c: float = 2.0
    b_c: float = 0.125

    # Kappa(q) — defaults to the paper's c1 * sqrt(log(e^{-c2*q} + 1))
    kappa: Callable[[float], float] = field(
        default_factory=lambda: lambda q: 1000.0 * (np.log(np.exp(-0.01 * q) + 1.0) ** 0.5)
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


def run(cfg: AggressiveImpactConfig) -> dict:
    if cfg.model == "hybrid" and cfg.bar_kappa is None:
        raise ValueError("hybrid model requires bar_kappa")

    hawkes = _native.MultiExponentialHawkes.with_stationary_state(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
    )
    market = _native.simulate_hawkes_as_market_orders(hawkes, cfg.time_horizon, cfg.seed)
    process = _native.AffineQueueProcess.new_queue(
        float(cfg.initial_queue_size), cfg.a_l, cfg.b_l, cfg.a_c, cfg.b_c,
    )
    q_events = _native.simulate_with_externals(process, cfg.time_horizon, market, cfg.seed)
    q_full = _native.merge_events(q_events, market)
    cond_by_dim = [list(arr) for arr in _native.extract_events_by_dim(q_events, 3, 2)]

    meta = _make_meta_orders(cfg)
    market_times = list(market.times())
    meta_times = list(meta.times())

    eval_entries = sorted([(t, True) for t in market_times] + [(t, False) for t in meta_times])
    eval_times = np.array([t for t, _ in eval_entries], dtype=np.float64)
    is_market_order = [b for _, b in eval_entries]
    bar_q_external = _native.merge_events(meta, market)

    q_at_eval = _native.sample_queue_at_times(q_full, cfg.initial_queue_size, eval_times)

    n_times = len(eval_times)
    queue_paths = np.empty((n_times, cfg.n_simulations + 1), dtype=np.uint32)
    queue_paths[:, 0] = q_at_eval
    impact_paths = np.empty((n_times, cfg.n_simulations), dtype=np.float64)

    for sim_idx in range(cfg.n_simulations):
        ctx = _native.ConditionalSimulationContext(
            process, cond_by_dim,
            cfg.time_horizon,
            cond_externals=market,
            new_externals=bar_q_external,
        )
        bar_q = ctx.simulate_queue_at_times(eval_times, cfg.initial_queue_size, seed=sim_idx)
        queue_paths[:, sim_idx + 1] = bar_q

        if cfg.model == "hybrid":
            raise NotImplementedError(
                "hybrid model wiring is added in Task 15; for now use model='propagator'."
            )

        result = _native.aggressive_impact_from_queue_samples(
            q_samples=q_at_eval, q_bar_samples=bar_q,
            eval_times=eval_times, is_market_order=is_market_order,
            hawkes=hawkes, kappa=cfg.kappa,
        )
        impact_paths[:, sim_idx] = result.impact()

    return {
        "times": eval_times,
        "queue_paths": queue_paths,
        "impact_paths": impact_paths,
        "event_types": np.array([1.0 if b else 0.0 for b in is_market_order]),
    }


def save(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
