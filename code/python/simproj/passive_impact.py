"""Facade for the passive impact experiment.

Wraps the bound primitives (Hawkes, AffineQueueProcess, ConditionalSimulationContext,
TailImpact) into a single `run(config)` entry point that returns the same arrays the
Rust binaries write today.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from . import _native


@dataclass
class PassiveImpactConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500
    initial_queue_size: int = 200
    mode: str = "single"            # "single" | "double"
    counterfactual: bool = False     # False: with us | True: without us

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
    return _native.create_meta_orders_from_times(times, target_dim=0, total_dims=3)


def _run_single_direction(cfg, direction: str) -> dict:
    """Run one direction ('with' or 'without') for single-queue mode.

    For direction='with', condition on q and simulate bar_q.
    For direction='without', condition on bar_q and simulate q.
    """
    hawkes = _make_hawkes(cfg)
    market = _native.simulate_hawkes_as_market_orders(hawkes, cfg.time_horizon, cfg.seed)
    process = _native.AffineQueueProcess.new_queue(
        float(cfg.initial_queue_size), cfg.a_l, cfg.b_l, cfg.a_c, cfg.b_c,
    )

    market_times = market.times()
    meta = _make_meta_orders(cfg)
    bar_q_externals = _native.merge_events(meta, market)
    q_externals = market

    n_times = len(market_times)
    queue_paths = np.empty((n_times, cfg.n_simulations + 1), dtype=np.uint32)
    impact_paths = np.zeros((n_times, cfg.n_simulations), dtype=np.float64)

    tail = _native.TailImpact.from_affine_queue(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
        cfg.b_l, cfg.b_c, list(market_times),
    )

    if direction == "with":
        q_internal = _native.simulate_with_externals(
            process, cfg.time_horizon, q_externals, cfg.seed,
        )
        q_full = _native.merge_events(q_internal, q_externals)
        cond_by_dim = [
            list(arr) for arr in _native.extract_events_by_dim(q_internal, 3, 2)
        ]
        queue_paths[:, 0] = _native.sample_queue_at_times(
            q_full, cfg.initial_queue_size, market_times,
        )
        ctx = _native.ConditionalSimulationContext(
            process, cond_by_dim,
            cfg.time_horizon,
            cond_externals=q_externals,
            new_externals=bar_q_externals,
        )

        for sim_idx in range(cfg.n_simulations):
            # ctx.simulate() already incorporates the new_externals (meta + market),
            # so we must NOT re-merge market here — doing so doubles dim=2 events.
            bar_q_events = ctx.simulate(seed=sim_idx)
            queue_paths[:, sim_idx + 1] = _native.sample_queue_at_times(
                bar_q_events, cfg.initial_queue_size, market_times,
            )
            impact_paths[:, sim_idx] = _native.compute_impact_path(
                q_full, bar_q_events, cfg.initial_queue_size, tail,
            )
    elif direction == "without":
        bar_q_internal = _native.simulate_with_externals(
            process, cfg.time_horizon, bar_q_externals, cfg.seed,
        )
        bar_q_full = _native.merge_events(bar_q_internal, bar_q_externals)
        cond_by_dim = [
            list(arr) for arr in _native.extract_events_by_dim(bar_q_internal, 3, 2)
        ]
        queue_paths[:, 0] = _native.sample_queue_at_times(
            bar_q_full, cfg.initial_queue_size, market_times,
        )
        ctx = _native.ConditionalSimulationContext(
            process, cond_by_dim,
            cfg.time_horizon,
            cond_externals=bar_q_externals,
            new_externals=q_externals,
        )

        for sim_idx in range(cfg.n_simulations):
            q_events = ctx.simulate(seed=sim_idx)
            queue_paths[:, sim_idx + 1] = _native.sample_queue_at_times(
                q_events, cfg.initial_queue_size, market_times,
            )
            impact_paths[:, sim_idx] = _native.compute_impact_path(
                q_events, bar_q_full, cfg.initial_queue_size, tail,
            )
    else:
        raise ValueError(f"direction must be 'with' or 'without'; got {direction!r}")

    return {
        "times": np.asarray(market_times, dtype=np.float64),
        "queue_paths": queue_paths,
        "impact_paths": impact_paths,
    }


def run(config: PassiveImpactConfig) -> dict:
    """Run the passive impact experiment per `config`.

    Valid values:
        counterfactual: False for with-us, True for without-us
        mode: 'single' (double-queue is deferred)
    """
    if config.mode not in ("single", "double"):
        raise ValueError(f"mode must be 'single' or 'double'; got {config.mode!r}")
    if config.mode == "double":
        raise NotImplementedError("double-queue facade — wired in a follow-up")
    return _run_single_direction(config, "without" if config.counterfactual else "with")


def save(result: dict, output_dir: str) -> None:
    """Persist a result dict as .npy files under output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    if "with" in result and "without" in result:
        save(result["with"], os.path.join(output_dir, "with"))
        save(result["without"], os.path.join(output_dir, "without"))
        return
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
