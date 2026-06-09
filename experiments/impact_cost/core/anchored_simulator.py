"""Anchored conditional queue simulation for empirical queue snapshots.

This module is experiment-local. It treats the observed queue snapshots as the
conditioning path and simulates only a displacement from that path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .cost_utils import CANCEL, LIMIT, MARKET, event_seconds
from .level_execution import market_side_for_queue


LIMIT_DIM = 0
CANCEL_DIM = 1
MARKET_DIM = 2


@dataclass(frozen=True)
class AnchoredSimulationResult:
    """Output of anchored queue simulations.

    `grid` is the evaluation grid in seconds. The queue arrays are aligned to
    that grid: `factual_queue` is the empirical post-event queue,
    `mechanical_no_us_queue` removes flagged passive L rows mechanically,
    `simulated_queues` has shape `(len(grid), n_simulations)`, and
    `simulated_offsets` is `q - bar_q`. `anchored_events` is the row-level input
    sent to Rust; `simulated_events` is the regrouped simulated event table
    returned for diagnostics.
    """

    grid: np.ndarray
    factual_queue: np.ndarray
    mechanical_no_us_queue: np.ndarray
    simulated_queues: np.ndarray
    simulated_offsets: np.ndarray
    simulated_events: pd.DataFrame
    anchored_events: pd.DataFrame


def build_anchored_events(
    window: pd.DataFrame,
    passive_l_flags: Iterable[bool],
    *,
    raw_side: str,
    queue_col: str,
    market_side: str | None = None,
    initial_q: int,
    origin: object | None = None,
    qty_col: str = "qty",
    ts_col: str = "ts",
    order_type_col: str = "order_type",
    side_col: str = "side",
) -> pd.DataFrame:
    """Build a row-level event table with observed pre/post queue snapshots.

    `bar_q_pre` is the empirical queue just before the row. Since the raw
    columns are post-event snapshots, it is the previous row's snapshot, except
    for the first row where `initial_q` is used.
    """
    flags = np.asarray(list(passive_l_flags), dtype=bool)
    if flags.shape != (len(window),):
        raise ValueError("passive_l_flags must have one boolean per dataframe row")

    seconds = event_seconds(window, ts_col=ts_col, origin=origin)
    types = window[order_type_col].astype(str).str.lower().to_numpy()
    sides = window[side_col].astype(str).to_numpy()
    qty = window[qty_col].to_numpy(dtype=np.int64)
    bar_post = window[queue_col].to_numpy(dtype=np.float64)
    if np.any(qty < 0):
        raise ValueError("qty must be non-negative")

    bar_pre = np.empty(len(window), dtype=np.float64)
    if len(window) > 0:
        bar_pre[0] = float(initial_q)
        bar_pre[1:] = bar_post[:-1]

    posting_side_label = str(raw_side).upper()
    consuming_side = market_side_for_queue(
        raw_side=raw_side,
        queue_col=queue_col,
        market_side=market_side,
    )
    dim = np.full(len(window), -1, dtype=np.int64)
    posting_side = sides == posting_side_label
    market_consuming_side = sides == consuming_side
    dim[(types == LIMIT) & posting_side] = LIMIT_DIM
    dim[(types == CANCEL) & posting_side] = CANCEL_DIM
    dim[(types == MARKET) & market_consuming_side] = MARKET_DIM

    is_passive_ours = flags & (types == LIMIT) & posting_side
    return pd.DataFrame(
        {
            "row_pos": np.arange(len(window), dtype=np.int64),
            "time": seconds,
            "ts": window[ts_col].astype(str).to_numpy(),
            "order_type": types,
            "side": sides,
            "dim": dim,
            "qty": qty,
            "bar_q_pre": bar_pre,
            "bar_q_post": bar_post,
            "is_passive_ours": is_passive_ours,
        }
    )


def event_dims_for_side(
    window: pd.DataFrame,
    *,
    raw_side: str,
    market_side: str | None = None,
    level_col: str | None = None,
    target_level: object = 1,
    order_type_col: str = "order_type",
    side_col: str = "side",
) -> np.ndarray:
    """Return native event dimensions for one queue side.

    Rows outside the selected side or level get dimension `-1`; selected rows
    get limit/cancel/market dimensions `0/1/2`.
    """
    types = window[order_type_col].astype(str).str.lower().to_numpy()
    sides = window[side_col].astype(str).to_numpy()
    dim = np.full(len(window), -1, dtype=np.int32)
    posting_side_label = str(raw_side).upper()
    consuming_side_label = (
        posting_side_label if market_side is None else str(market_side).upper()
    )
    posting_side = sides == posting_side_label
    consuming_side = sides == consuming_side_label
    if level_col is not None:
        if level_col not in window.columns:
            raise KeyError(f"level column {level_col!r} is not present")
        level_mask = (window[level_col] == target_level).to_numpy(dtype=bool)
        posting_side &= level_mask
        consuming_side &= level_mask
    dim[(types == LIMIT) & posting_side] = LIMIT_DIM
    dim[(types == CANCEL) & posting_side] = CANCEL_DIM
    dim[(types == MARKET) & consuming_side] = MARKET_DIM
    return dim


def select_passive_limit_flags(
    window: pd.DataFrame,
    policy: str,
    *,
    raw_side: str,
    market_side: str | None = None,
    level_col: str | None = None,
    target_level: object = 1,
    every_seconds: float | None = None,
    fraction: float | None = None,
    indices: Iterable[int] | None = None,
    index_base: int = 1,
    seed: int | None = None,
    origin: object | None = None,
    ts_col: str = "ts",
) -> np.ndarray:
    """Select passive L rows with the native Rust policy helpers.

    Accepted policies are `none`, `first_every`, `random_fraction`, and
    `indices`. The returned boolean array has one entry per input row and is
    intended to be passed to `simulate_anchored_queue_paths`.
    """
    policy = policy.lower()
    if policy == "none":
        return np.zeros(len(window), dtype=bool)

    simproj = _import_simproj()
    times = event_seconds(window, ts_col=ts_col, origin=origin)
    dims = event_dims_for_side(
        window,
        raw_side=raw_side,
        market_side=market_side,
        level_col=level_col,
        target_level=target_level,
    )

    if policy == "first_every":
        if every_seconds is None:
            raise ValueError("every_seconds must be provided for first_every")
        return np.asarray(
            simproj.select_limit_flags_first_every(
                times.astype(np.float64),
                dims.astype(np.int32),
                float(every_seconds),
            ),
            dtype=bool,
        )

    if policy == "random_fraction":
        if fraction is None:
            raise ValueError("fraction must be provided for random_fraction")
        return np.asarray(
            simproj.select_limit_flags_random_fraction(
                dims.astype(np.int32),
                float(fraction),
                None if seed is None else int(seed),
            ),
            dtype=bool,
        )

    if policy == "indices":
        if indices is None:
            raise ValueError("indices must be provided for indices policy")
        return np.asarray(
            simproj.select_limit_flags_indices(
                dims.astype(np.int32),
                [int(idx) for idx in indices],
                int(index_base),
            ),
            dtype=bool,
        )

    raise ValueError("policy must be 'none', 'first_every', 'random_fraction', or 'indices'")


def simulate_anchored_queue_paths(
    window: pd.DataFrame,
    passive_l_flags: Iterable[bool],
    *,
    raw_side: str,
    queue_col: str,
    market_side: str | None = None,
    initial_q: int,
    horizon_seconds: float,
    grid: np.ndarray,
    n_simulations: int,
    seed: int | None,
    a_l: float,
    b_l: float,
    a_c: float,
    b_c: float,
    origin: object | None = None,
) -> AnchoredSimulationResult:
    """Simulate counterfactual queues anchored on empirical snapshots.

    The observed queue `bar_q` is exogenous. Rust simulates `dq = q - bar_q`;
    intensities are evaluated at `bar_q_pre + dq`.
    """
    simproj = _import_simproj()
    anchored_events = build_anchored_events(
        window,
        passive_l_flags,
        raw_side=raw_side,
        queue_col=queue_col,
        market_side=market_side,
        initial_q=initial_q,
        origin=origin,
    )
    native = simproj.simulate_anchored_affine_queue(
        anchored_events["time"].to_numpy(dtype=np.float64),
        anchored_events["dim"].to_numpy(dtype=np.int32),
        anchored_events["qty"].to_numpy(dtype=np.uint32),
        anchored_events["bar_q_pre"].to_numpy(dtype=np.float64),
        anchored_events["bar_q_post"].to_numpy(dtype=np.float64),
        anchored_events["is_passive_ours"].to_numpy(dtype=np.bool_),
        np.asarray(grid, dtype=np.float64),
        float(initial_q),
        float(horizon_seconds),
        int(n_simulations),
        float(a_l),
        float(b_l),
        float(a_c),
        float(b_c),
        None if seed is None else int(seed),
    )

    n_times = int(native["n_times"])
    n_sims = int(native["n_simulations"])
    factual_queue = np.asarray(native["factual_queue"], dtype=np.float64)
    mechanical_no_us_queue = np.asarray(native["mechanical_queue"], dtype=np.float64)
    simulated_queues = np.asarray(native["queue_samples"], dtype=np.float64).reshape(
        n_times, n_sims
    )
    simulated_offsets = np.asarray(native["offset_samples"], dtype=np.float64).reshape(
        n_times, n_sims
    )
    simulated_events = _regroup_native_events(
        times=np.asarray(native["event_times"], dtype=np.float64),
        dims=np.asarray(native["event_dims"], dtype=np.int64),
        qtys=np.asarray(native["event_qtys"], dtype=np.int64),
        simulations=np.asarray(native["event_simulations"], dtype=np.int64),
    )

    return AnchoredSimulationResult(
        grid=grid,
        factual_queue=factual_queue,
        mechanical_no_us_queue=mechanical_no_us_queue,
        simulated_queues=simulated_queues,
        simulated_offsets=simulated_offsets,
        simulated_events=simulated_events,
        anchored_events=anchored_events,
    )


def _import_simproj():
    try:
        import simproj  # type: ignore

        return simproj
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        code_python = repo_root / "code" / "python"
        if str(code_python) not in sys.path:
            sys.path.insert(0, str(code_python))
        import simproj  # type: ignore

        return simproj


def _regroup_native_events(
    *,
    times: np.ndarray,
    dims: np.ndarray,
    qtys: np.ndarray,
    simulations: np.ndarray,
) -> pd.DataFrame:
    if times.size == 0:
        return pd.DataFrame(
            {
                "simulation": pd.Series(dtype=np.int64),
                "time": pd.Series(dtype=np.float64),
                "dim": pd.Series(dtype=np.int64),
                "qty": pd.Series(dtype=np.int64),
            }
        )
    flat = pd.DataFrame(
        {"simulation": simulations, "time": times, "dim": dims, "qty": qtys}
    )
    grouped = (
        flat.groupby(["simulation", "time", "dim"], sort=True)
        ["qty"]
        .sum()
        .rename("qty")
        .reset_index()
    )
    grouped["simulation"] = grouped["simulation"].astype(np.int64)
    grouped["dim"] = grouped["dim"].astype(np.int64)
    grouped["qty"] = grouped["qty"].astype(np.int64)
    return grouped[["simulation", "time", "dim", "qty"]]
