"""Synthetic passive order lifecycle policies.

This module deliberately stays independent from price-impact math. It only
generates own passive order events:

- posts add displayed passive quantity;
- fills remove displayed quantity and create execution-cost timestamps;
- cancels remove displayed quantity without direct execution cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


FillCountModel = Literal["binomial", "fixed"]
FillTimeModel = Literal["clustered_exponential", "independent_exponential"]
FillSelection = Literal["oldest", "random"]


@dataclass(frozen=True)
class PassiveLifecycleConfig:
    """Configuration for repeated passive post/fill/cancel cycles."""

    n_cycles: int = 3
    orders_per_cycle: int = 10
    order_qty: int = 1
    posting_spacing_seconds: float = 0.010
    fill_count_model: FillCountModel = "binomial"
    fill_probability: float = 1.0 / 7.0
    fixed_filled_orders: int | None = None
    fill_selection: FillSelection = "oldest"
    fill_time_model: FillTimeModel = "clustered_exponential"
    fill_wait_mean_seconds: float = 0.150
    fill_gap_mean_seconds: float = 0.010
    min_resting_seconds: float = 0.300
    cancel_delay_seconds: float = 0.010
    cancel_jitter_seconds: float = 0.001
    repost_delay_seconds: float = 0.050


def validate_passive_lifecycle_config(cfg: PassiveLifecycleConfig) -> None:
    """Validate lifecycle policy parameters."""
    if cfg.n_cycles <= 0:
        raise ValueError("n_cycles must be positive")
    if cfg.orders_per_cycle <= 0:
        raise ValueError("orders_per_cycle must be positive")
    if cfg.order_qty <= 0:
        raise ValueError("order_qty must be positive")
    if cfg.posting_spacing_seconds <= 0.0:
        raise ValueError("posting_spacing_seconds must be positive")
    if cfg.fill_count_model not in {"binomial", "fixed"}:
        raise ValueError("fill_count_model must be 'binomial' or 'fixed'")
    if not 0.0 <= cfg.fill_probability <= 1.0:
        raise ValueError("fill_probability must be in [0, 1]")
    if cfg.fixed_filled_orders is not None and cfg.fixed_filled_orders < 0:
        raise ValueError("fixed_filled_orders must be nonnegative or None")
    if cfg.fill_selection not in {"oldest", "random"}:
        raise ValueError("fill_selection must be 'oldest' or 'random'")
    if cfg.fill_time_model not in {"clustered_exponential", "independent_exponential"}:
        raise ValueError(
            "fill_time_model must be 'clustered_exponential' or "
            "'independent_exponential'"
        )
    if cfg.fill_wait_mean_seconds <= 0.0:
        raise ValueError("fill_wait_mean_seconds must be positive")
    if cfg.fill_gap_mean_seconds <= 0.0:
        raise ValueError("fill_gap_mean_seconds must be positive")
    if cfg.min_resting_seconds < 0.0:
        raise ValueError("min_resting_seconds must be nonnegative")
    if cfg.cancel_delay_seconds < 0.0:
        raise ValueError("cancel_delay_seconds must be nonnegative")
    if cfg.cancel_jitter_seconds < 0.0:
        raise ValueError("cancel_jitter_seconds must be nonnegative")
    if cfg.repost_delay_seconds < 0.0:
        raise ValueError("repost_delay_seconds must be nonnegative")


def generate_passive_lifecycle(
    cfg: PassiveLifecycleConfig,
    *,
    seed: int | None = None,
    episode_id: int = 0,
    policy_path_id: int = 0,
    start_time_s: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Generate one synthetic passive lifecycle path.

    Returns a dict with `orders`, `fills`, `cancels`, `events`, and
    `cycle_summary` data frames. Time columns are seconds from the episode
    origin.
    """
    validate_passive_lifecycle_config(cfg)
    rng = np.random.default_rng(seed)
    cycle_start = float(start_time_s)
    next_order_id = 0
    order_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    cancel_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []

    for cycle_id in range(int(cfg.n_cycles)):
        cycle_order_ids = np.arange(
            next_order_id,
            next_order_id + int(cfg.orders_per_cycle),
            dtype=np.int64,
        )
        slots = np.arange(int(cfg.orders_per_cycle), dtype=np.int64)
        post_times = cycle_start + slots.astype(np.float64) * float(
            cfg.posting_spacing_seconds
        )
        posting_end = cycle_start + float(cfg.orders_per_cycle) * float(
            cfg.posting_spacing_seconds
        )

        n_filled = _draw_fill_count(cfg, rng)
        filled_slots = _select_filled_slots(cfg, rng, n_filled)
        fill_times = _draw_fill_times(cfg, rng, post_times, posting_end, filled_slots)
        filled_by_slot = {
            int(slot): float(fill_time)
            for slot, fill_time in zip(filled_slots.astype(int), fill_times)
        }

        last_fill_time = max(fill_times) if len(fill_times) else np.nan
        cancel_base_time = max(
            posting_end + float(cfg.min_resting_seconds),
            (
                float(last_fill_time) + float(cfg.cancel_delay_seconds)
                if np.isfinite(last_fill_time)
                else -np.inf
            ),
        )
        unfilled_slots = np.asarray(
            [int(slot) for slot in slots if int(slot) not in filled_by_slot],
            dtype=np.int64,
        )
        cancel_times = _draw_cancel_times(cfg, rng, cancel_base_time, len(unfilled_slots))
        cancel_by_slot = {
            int(slot): float(cancel_time)
            for slot, cancel_time in zip(unfilled_slots.astype(int), cancel_times)
        }

        for slot, order_id, post_time in zip(slots, cycle_order_ids, post_times):
            slot_int = int(slot)
            order_id_int = int(order_id)
            fill_time = filled_by_slot.get(slot_int, np.nan)
            cancel_time = cancel_by_slot.get(slot_int, np.nan)
            terminal_action = "fill" if np.isfinite(fill_time) else "cancel"
            terminal_time = fill_time if np.isfinite(fill_time) else cancel_time
            base = {
                "episode_id": int(episode_id),
                "policy_path_id": int(policy_path_id),
                "cycle_id": int(cycle_id),
                "order_id": order_id_int,
                "order_slot": slot_int,
                "post_time_s": float(post_time),
                "qty": int(cfg.order_qty),
            }
            order_rows.append(
                base
                | {
                    "filled": bool(np.isfinite(fill_time)),
                    "fill_time_s": float(fill_time) if np.isfinite(fill_time) else np.nan,
                    "cancel_time_s": (
                        float(cancel_time) if np.isfinite(cancel_time) else np.nan
                    ),
                    "terminal_time_s": (
                        float(terminal_time) if np.isfinite(terminal_time) else np.nan
                    ),
                    "terminal_action": terminal_action,
                }
            )
            event_rows.append(
                base
                | {
                    "time_s": float(post_time),
                    "event_kind": "post",
                    "displacement_delta": int(cfg.order_qty),
                }
            )
            if np.isfinite(fill_time):
                fill_row = base | {"fill_time_s": float(fill_time)}
                fill_rows.append(fill_row)
                event_rows.append(
                    base
                    | {
                        "time_s": float(fill_time),
                        "event_kind": "fill",
                        "displacement_delta": -int(cfg.order_qty),
                    }
                )
            elif np.isfinite(cancel_time):
                cancel_row = base | {"cancel_time_s": float(cancel_time)}
                cancel_rows.append(cancel_row)
                event_rows.append(
                    base
                    | {
                        "time_s": float(cancel_time),
                        "event_kind": "cancel",
                        "displacement_delta": -int(cfg.order_qty),
                    }
                )

        cycle_end = _cycle_end_time(
            posting_end=posting_end,
            fill_times=fill_times,
            cancel_times=cancel_times,
        )
        cycle_rows.append(
            {
                "episode_id": int(episode_id),
                "policy_path_id": int(policy_path_id),
                "cycle_id": int(cycle_id),
                "cycle_start_time_s": float(cycle_start),
                "posting_end_time_s": float(posting_end),
                "cycle_end_time_s": float(cycle_end),
                "n_posted_orders": int(cfg.orders_per_cycle),
                "n_filled_orders": int(n_filled),
                "n_canceled_orders": int(cfg.orders_per_cycle) - int(n_filled),
            }
        )
        next_order_id += int(cfg.orders_per_cycle)
        cycle_start = float(cycle_end) + float(cfg.repost_delay_seconds)

    orders = pd.DataFrame(order_rows)
    fills = pd.DataFrame(fill_rows)
    cancels = pd.DataFrame(cancel_rows)
    events = pd.DataFrame(event_rows)
    cycle_summary = pd.DataFrame(cycle_rows)
    return {
        "orders": _ensure_order_columns(orders),
        "fills": _ensure_fill_columns(fills),
        "cancels": _ensure_cancel_columns(cancels),
        "events": _ensure_event_columns(events),
        "cycle_summary": _ensure_cycle_columns(cycle_summary),
    }


def active_displacement_at_times(
    events: pd.DataFrame,
    times: np.ndarray,
    *,
    include_events_at_time: bool = True,
) -> np.ndarray:
    """Return active displayed own quantity at the requested times."""
    grid = np.asarray(times, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if events.empty:
        return np.zeros(grid.shape, dtype=np.float64)
    grouped = (
        events.groupby("time_s", sort=True)["displacement_delta"]
        .sum()
        .rename("delta")
        .reset_index()
    )
    event_times = grouped["time_s"].to_numpy(dtype=np.float64)
    cumulative = np.cumsum(grouped["delta"].to_numpy(dtype=np.float64))
    side = "right" if include_events_at_time else "left"
    idx = np.searchsorted(event_times, grid, side=side) - 1
    out = np.zeros(grid.shape, dtype=np.float64)
    valid = idx >= 0
    out[valid] = cumulative[idx[valid]]
    return np.maximum(out, 0.0)


def _draw_fill_count(cfg: PassiveLifecycleConfig, rng: np.random.Generator) -> int:
    if cfg.fill_count_model == "fixed":
        n = int(cfg.orders_per_cycle if cfg.fixed_filled_orders is None else cfg.fixed_filled_orders)
    else:
        n = int(rng.binomial(int(cfg.orders_per_cycle), float(cfg.fill_probability)))
    return int(np.clip(n, 0, int(cfg.orders_per_cycle)))


def _select_filled_slots(
    cfg: PassiveLifecycleConfig,
    rng: np.random.Generator,
    n_filled: int,
) -> np.ndarray:
    if n_filled <= 0:
        return np.array([], dtype=np.int64)
    slots = np.arange(int(cfg.orders_per_cycle), dtype=np.int64)
    if cfg.fill_selection == "oldest":
        return slots[: int(n_filled)]
    return np.sort(rng.choice(slots, size=int(n_filled), replace=False).astype(np.int64))


def _draw_fill_times(
    cfg: PassiveLifecycleConfig,
    rng: np.random.Generator,
    post_times: np.ndarray,
    posting_end: float,
    filled_slots: np.ndarray,
) -> np.ndarray:
    n_filled = len(filled_slots)
    if n_filled <= 0:
        return np.array([], dtype=np.float64)
    if cfg.fill_time_model == "clustered_exponential":
        first_delay = rng.exponential(float(cfg.fill_wait_mean_seconds))
        if n_filled == 1:
            gaps = np.array([], dtype=np.float64)
        else:
            gaps = rng.exponential(float(cfg.fill_gap_mean_seconds), size=n_filled - 1)
        times = float(posting_end) + first_delay + np.concatenate(
            [np.array([0.0], dtype=np.float64), np.cumsum(gaps)]
        )
        return np.sort(times.astype(np.float64))

    latencies = rng.exponential(float(cfg.fill_wait_mean_seconds), size=n_filled)
    raw_times = post_times[filled_slots.astype(int)] + latencies
    return np.sort(raw_times.astype(np.float64))


def _draw_cancel_times(
    cfg: PassiveLifecycleConfig,
    rng: np.random.Generator,
    cancel_base_time: float,
    n_cancels: int,
) -> np.ndarray:
    if n_cancels <= 0:
        return np.array([], dtype=np.float64)
    if cfg.cancel_jitter_seconds == 0.0:
        return np.full(n_cancels, float(cancel_base_time), dtype=np.float64)
    jitter = rng.uniform(0.0, float(cfg.cancel_jitter_seconds), size=n_cancels)
    return np.sort(float(cancel_base_time) + jitter)


def _cycle_end_time(
    *,
    posting_end: float,
    fill_times: np.ndarray,
    cancel_times: np.ndarray,
) -> float:
    candidates = [float(posting_end)]
    if len(fill_times):
        candidates.append(float(np.max(fill_times)))
    if len(cancel_times):
        candidates.append(float(np.max(cancel_times)))
    return max(candidates)


def _ensure_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "episode_id": "int64",
        "policy_path_id": "int64",
        "cycle_id": "int64",
        "order_id": "int64",
        "order_slot": "int64",
        "post_time_s": "float64",
        "qty": "int64",
        "filled": "bool",
        "fill_time_s": "float64",
        "cancel_time_s": "float64",
        "terminal_time_s": "float64",
        "terminal_action": "object",
    }
    return _ensure_columns(df, columns)


def _ensure_fill_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "episode_id": "int64",
        "policy_path_id": "int64",
        "cycle_id": "int64",
        "order_id": "int64",
        "order_slot": "int64",
        "post_time_s": "float64",
        "qty": "int64",
        "fill_time_s": "float64",
    }
    return _ensure_columns(df, columns)


def _ensure_cancel_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "episode_id": "int64",
        "policy_path_id": "int64",
        "cycle_id": "int64",
        "order_id": "int64",
        "order_slot": "int64",
        "post_time_s": "float64",
        "qty": "int64",
        "cancel_time_s": "float64",
    }
    return _ensure_columns(df, columns)


def _ensure_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "episode_id": "int64",
        "policy_path_id": "int64",
        "cycle_id": "int64",
        "order_id": "int64",
        "order_slot": "int64",
        "post_time_s": "float64",
        "qty": "int64",
        "time_s": "float64",
        "event_kind": "object",
        "displacement_delta": "int64",
    }
    return _ensure_columns(df, columns)


def _ensure_cycle_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "episode_id": "int64",
        "policy_path_id": "int64",
        "cycle_id": "int64",
        "cycle_start_time_s": "float64",
        "posting_end_time_s": "float64",
        "cycle_end_time_s": "float64",
        "n_posted_orders": "int64",
        "n_filled_orders": "int64",
        "n_canceled_orders": "int64",
    }
    return _ensure_columns(df, columns)


def _ensure_columns(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in columns.items():
        if col not in out.columns:
            out[col] = pd.Series(dtype=dtype)
    out = out[list(columns)]
    for col, dtype in columns.items():
        if len(out):
            out[col] = out[col].astype(dtype)
    return out
