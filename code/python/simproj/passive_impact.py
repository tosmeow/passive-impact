"""Facade for the passive impact experiment.

Wraps the bound primitives (Hawkes, AffineQueueProcess, ConditionalSimulationContext,
TailImpact) into a single `run(config)` entry point that returns the same arrays the
Rust binaries write today.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from . import _native


@dataclass
class PassiveImpactConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500
    initial_queue_size: int = 200
    mode: str = "single"            # "single" | "double"
    side: str = "both"              # "with" | "without" | "both"

    # Hawkes
    mu: float = 1.0
    alpha: list = field(default_factory=lambda: [0.065, 0.2, 0.325, 0.65])
    beta: list = field(default_factory=lambda: [0.15, 0.60, 2.5, 10.0])

    # Affine queue
    a_l: float = 100.0
    b_l: float = -0.275
    a_c: float = 2.0
    b_c: float = 0.125

    # Metaorder: int → evenly-spaced inside metaorder_window;
    #            list[float] / np.ndarray → explicit times (window ignored)
    metaorder: Union[int, list, np.ndarray] = 375
    metaorder_window: tuple = (1.0, 80.0)

    seed: int = 42


def _make_hawkes(cfg: PassiveImpactConfig):
    return _native.MultiExponentialHawkes.with_stationary_state(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
    )


def _make_meta_orders(cfg: PassiveImpactConfig):
    if isinstance(cfg.metaorder, int):
        return _native.create_meta_orders(cfg.metaorder, *cfg.metaorder_window)
    times = np.asarray(cfg.metaorder, dtype=np.float64)
    return _native.create_meta_orders_from_times(times, target_dim=2, total_dims=3)


def _run_single_side(cfg, side: str) -> dict:
    """Run one side ('with' or 'without') for single-queue mode.

    Note: this initial cut returns impact_paths filled with zeros — the real
    impact computation is wired in Task 13 (next).
    """
    hawkes = _make_hawkes(cfg)
    market = _native.simulate_hawkes_as_market_orders(hawkes, cfg.time_horizon, cfg.seed)
    process = _native.AffineQueueProcess.new_queue(
        float(cfg.initial_queue_size), cfg.a_l, cfg.b_l, cfg.a_c, cfg.b_c,
    )
    q_events = _native.simulate_with_externals(process, cfg.time_horizon, market, cfg.seed)
    full_q = _native.merge_events(q_events, market)

    market_times = market.times()
    cond_by_dim = [list(arr) for arr in _native.extract_events_by_dim(q_events, 3, 2)]

    meta = _make_meta_orders(cfg)
    bar_q_externals = _native.merge_events(meta, market)

    ctx = _native.ConditionalSimulationContext(
        process, cond_by_dim,
        cfg.time_horizon,
        cond_externals=market,
        new_externals=bar_q_externals,
    )

    # NOTE: the test harness for this task asserts shape only, since impact
    # is wired in Task 13. Placeholder zeros here.
    n_times = len(market_times)
    queue_paths = np.empty((n_times, cfg.n_simulations + 1), dtype=np.uint32)
    impact_paths = np.zeros((n_times, cfg.n_simulations), dtype=np.float64)
    q_at_market = _native.sample_queue_at_times(full_q, cfg.initial_queue_size, market_times)
    queue_paths[:, 0] = q_at_market

    for sim_idx in range(cfg.n_simulations):
        bar_q_samples = ctx.simulate_queue_at_times(
            market_times, cfg.initial_queue_size, seed=sim_idx,
        )
        queue_paths[:, sim_idx + 1] = bar_q_samples

    return {
        "times": np.asarray(market_times, dtype=np.float64),
        "queue_paths": queue_paths,
        "impact_paths": impact_paths,
    }


def run(config: PassiveImpactConfig) -> dict:
    """Run the passive impact experiment per `config`.

    For mode='single' + side='both', returns a dict with keys
    {'with': {...}, 'without': {...}} each holding the per-side result dict.
    """
    if config.mode == "double":
        raise NotImplementedError("double-queue facade — wired in a follow-up")
    if config.side == "both":
        return {
            "with": _run_single_side(config, "with"),
            "without": _run_single_side(config, "without"),
        }
    return _run_single_side(config, config.side)


def save(result: dict, output_dir: str) -> None:
    """Persist a result (or both-side dict) as .npy files under output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    if "with" in result and "without" in result:
        save(result["with"], os.path.join(output_dir, "with"))
        save(result["without"], os.path.join(output_dir, "without"))
        return
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
