"""Facade for queue-only counterfactual simulation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from . import _native


@dataclass
class QueueSimulationConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500
    n_eval_times: int = 1000
    initial_queue_size: int = 200
    mode: str = "single"            # "single" | "double"

    mu: float = 1.0
    alpha: list = field(default_factory=lambda: [0.065, 0.2, 0.325, 0.65])
    beta: list = field(default_factory=lambda: [0.15, 0.60, 2.5, 10.0])

    a_l: float = 100.0
    b_l: float = -0.275
    a_c: float = 2.0
    b_c: float = 0.125

    metaorder: Union[int, list, np.ndarray] = 375
    metaorder_window: tuple = (1.0, 80.0)

    seed: int = 42


def run(cfg: QueueSimulationConfig) -> dict:
    if cfg.mode == "double":
        raise NotImplementedError("double-queue queue_simulation — follow-up")

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

    if isinstance(cfg.metaorder, int):
        meta = _native.create_meta_orders(cfg.metaorder, *cfg.metaorder_window)
    else:
        meta = _native.create_meta_orders_from_times(
            np.asarray(cfg.metaorder, dtype=np.float64), target_dim=2, total_dims=3,
        )
    bar_q_external = _native.merge_events(meta, market)

    times = np.linspace(0.0, cfg.time_horizon, cfg.n_eval_times).astype(np.float64)
    q_at_times = _native.sample_queue_at_times(q_full, cfg.initial_queue_size, times)

    queue_paths = np.empty((cfg.n_eval_times, cfg.n_simulations + 1), dtype=np.uint32)
    queue_paths[:, 0] = q_at_times
    for sim_idx in range(cfg.n_simulations):
        ctx = _native.ConditionalSimulationContext(
            process, cond_by_dim,
            cfg.time_horizon,
            cond_externals=market,
            new_externals=bar_q_external,
        )
        queue_paths[:, sim_idx + 1] = ctx.simulate_queue_at_times(
            times, cfg.initial_queue_size, seed=sim_idx,
        )

    return {"times": times, "queue_paths": queue_paths}


def save(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
