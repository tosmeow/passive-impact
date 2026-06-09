"""Selection utilities for passive execution latency grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LatencyFilterConfig:
    """Filter for passive orders/windows entering impact-cost runs.

    `max_latency_seconds` and `require_completed` define the order-level
    filter. `selection_mode` controls how those selected orders lift to
    windows: `orders`, `window_any`, `window_at_least`, or `window_all`.
    `required_slots` refers to zero-based order slots from the latency grid.
    """

    max_latency_seconds: float | None = 30.0
    require_completed: bool = True
    selection_mode: str = "orders"
    min_orders: int = 1
    required_slots: tuple[int, ...] = ()


def select_latency_orders(
    orders: pd.DataFrame,
    cfg: LatencyFilterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selected order rows and the corresponding window summary.

    Modes:
    - `orders`: keep every order satisfying the order-level filter.
    - `window_any`: same selected orders, but only windows with at least one selected order.
    - `window_at_least`: only windows with at least `min_orders` selected orders.
    - `window_all`: only windows where every required slot is selected. If
      `required_slots` is empty, this means slots `0, ..., min_orders - 1`.
    """
    required = {
        "window_id",
        "window_start",
        "order_id",
        "order_slot",
        "latency_s",
        "completed_time_s",
    }
    missing = required.difference(orders.columns)
    if missing:
        raise KeyError(f"orders dataframe is missing columns: {sorted(missing)}")

    mode = cfg.selection_mode.lower()
    if mode not in {"orders", "window_any", "window_at_least", "window_all"}:
        raise ValueError(
            "selection_mode must be 'orders', 'window_any', 'window_at_least', or 'window_all'"
        )
    if cfg.min_orders <= 0:
        raise ValueError("min_orders must be positive")

    selected = _order_level_filter(orders, cfg)
    if selected.empty:
        return selected.copy(), _empty_window_summary()

    if mode == "orders" or mode == "window_any":
        keep_windows = set(selected["window_id"].astype(int))
    elif mode == "window_at_least":
        counts = selected.groupby("window_id")["order_id"].count()
        keep_windows = set(counts[counts >= cfg.min_orders].index.astype(int))
    else:
        slots = _required_slots(cfg)
        slot_sets = selected.groupby("window_id")["order_slot"].apply(
            lambda x: set(int(v) for v in x)
        )
        keep_windows = {
            int(window_id)
            for window_id, window_slots in slot_sets.items()
            if slots.issubset(window_slots)
        }

    selected = selected[selected["window_id"].astype(int).isin(keep_windows)].copy()
    windows = _window_summary(selected)
    return selected.reset_index(drop=True), windows


def _order_level_filter(orders: pd.DataFrame, cfg: LatencyFilterConfig) -> pd.DataFrame:
    mask = pd.Series(True, index=orders.index)
    if cfg.require_completed:
        mask &= orders["completed_time_s"].notna()
    if cfg.max_latency_seconds is not None:
        mask &= orders["latency_s"].notna()
        mask &= orders["latency_s"] <= float(cfg.max_latency_seconds)
    if cfg.required_slots:
        slots = set(int(slot) for slot in cfg.required_slots)
        mask &= orders["order_slot"].astype(int).isin(slots)
    return orders[mask].copy()


def _required_slots(cfg: LatencyFilterConfig) -> set[int]:
    if cfg.required_slots:
        return set(int(slot) for slot in cfg.required_slots)
    return set(range(int(cfg.min_orders)))


def _window_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return _empty_window_summary()
    grouped = selected.groupby(["window_id", "window_start"], sort=True)
    summary = grouped.agg(
        n_selected_orders=("order_id", "count"),
        selected_slots=("order_slot", _format_slots),
        max_latency_s=("latency_s", "max"),
        mean_latency_s=("latency_s", "mean"),
        total_selected_qty=("post_qty", "sum") if "post_qty" in selected.columns else ("order_id", "count"),
    ).reset_index()
    return summary


def _empty_window_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": pd.Series(dtype=np.int64),
            "window_start": pd.Series(dtype=object),
            "n_selected_orders": pd.Series(dtype=np.int64),
            "selected_slots": pd.Series(dtype=object),
            "max_latency_s": pd.Series(dtype=np.float64),
            "mean_latency_s": pd.Series(dtype=np.float64),
            "total_selected_qty": pd.Series(dtype=np.float64),
        }
    )


def _format_slots(values: Iterable[int]) -> str:
    return ",".join(str(int(slot)) for slot in sorted(set(values)))
