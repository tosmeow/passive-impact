"""Minute-by-minute passive execution latency grid on raw first-level depth."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.level_execution import market_side_for_queue, q1_column_for_side


LIMIT_DIM = 0
CANCEL_DIM = 1
MARKET_DIM = 2
IGNORED_DIM = -1


@dataclass(frozen=True)
class ExecutionLatencyGridConfig:
    """Configuration for minute-by-minute passive fill latency measurement.

    The raw input must have `ts`, `order_type`, `side`, `qty`, and first-level
    depth columns (`a_1`/`b_1`). Each minute window posts up to `n_orders`
    inferred first-level passive limits spaced by `order_spacing_seconds`, then
    tracks fills for `tracking_horizon_seconds`.
    """

    raw_level_path: str = "experiments/impact_cost/load_experiments/data/raw/2025_05_29_ESM5.parquet"
    output_dir: str = "experiments/impact_cost/archive/diagnostics/data/execution_latency_grid"
    start_time: str | None = None
    end_time: str | None = None
    window_start_policy: str = "clock"
    minute_seconds: float = 60.0
    n_orders: int = 3
    order_spacing_seconds: float = 1.0
    tracking_horizon_seconds: float | None = 900.0
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    cancellation_policy: str = "top"
    theta: float = 1.0
    cap_position_by_queue_post: bool = False
    require_full_sequence: bool = False
    require_sequence_completion: bool = False
    exclude_queue_reset_windows: bool = False
    seed: int = 2027


def run_execution_latency_grid(cfg: ExecutionLatencyGridConfig) -> dict[str, Any]:
    """Run the raw first-level latency grid and write order/fill CSV outputs."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_raw_depth(cfg)
    arrays = _build_first_level_arrays(
        raw,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
    )
    starts = _window_starts(raw, arrays, cfg)
    if starts.empty:
        raise ValueError("no minute starts available for execution latency grid")

    simproj = _import_simproj()
    native = _run_native_latency_grid(simproj, arrays, cfg, starts)
    orders, fills = _native_latency_frames(native, arrays, starts)
    sequence_summary = _sequence_summary(orders, arrays, starts, cfg)
    orders = _annotate_orders_with_sequence_status(orders, sequence_summary, cfg)
    fills = _annotate_fills_with_sequence_status(fills, sequence_summary)
    summary = _summarize_orders(orders, starts)
    valid_orders = _valid_orders(orders)
    valid_fills = _valid_fills(fills)
    valid_sequence_summary = sequence_summary[sequence_summary["valid_sequence"]].copy()
    valid_slot_summary = _summarize_valid_order_latencies(valid_orders)

    orders_path = output_dir / "passive_execution_latencies_by_minute.csv"
    summary_path = output_dir / "passive_execution_minute_summary.csv"
    fills_path = output_dir / "passive_execution_fills_by_minute.csv"
    sequence_summary_path = output_dir / "passive_execution_sequence_summary.csv"
    valid_orders_path = output_dir / "passive_execution_valid_orders.csv"
    valid_fills_path = output_dir / "passive_execution_valid_fills.csv"
    valid_slot_summary_path = output_dir / "passive_execution_valid_slot_latency_summary.csv"
    orders.to_csv(orders_path, index=False)
    summary.to_csv(summary_path, index=False)
    fills.to_csv(fills_path, index=False)
    sequence_summary.to_csv(sequence_summary_path, index=False)
    valid_orders.to_csv(valid_orders_path, index=False)
    valid_fills.to_csv(valid_fills_path, index=False)
    valid_slot_summary.to_csv(valid_slot_summary_path, index=False)

    latency_plot = output_dir / "passive_execution_latency_by_minute.png"
    hist_plot = output_dir / "passive_execution_latency_histogram.png"
    _plot_latency_by_minute(latency_plot, orders, cfg)
    _plot_latency_histogram(hist_plot, orders)

    with open(output_dir / "execution_latency_grid_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg) | {"q1_col": arrays["q1_col"]}, f, indent=2)

    return {
        "output_dir": output_dir,
        "orders_path": orders_path,
        "summary_path": summary_path,
        "fills_path": fills_path,
        "sequence_summary_path": sequence_summary_path,
        "valid_orders_path": valid_orders_path,
        "valid_fills_path": valid_fills_path,
        "valid_slot_summary_path": valid_slot_summary_path,
        "latency_plot": latency_plot,
        "histogram_plot": hist_plot,
        "q1_col": arrays["q1_col"],
        "n_windows": int(len(starts)),
        "n_posted_orders": int(len(orders)),
        "n_completed_orders": int(orders["completed_time_s"].notna().sum()),
        "n_valid_sequence_windows": int(len(valid_sequence_summary)),
        "n_valid_completed_orders": int(valid_orders["completed_before_reset"].sum()) if len(valid_orders) else 0,
        "completion_rate": float(orders["completed_time_s"].notna().mean()) if len(orders) else 0.0,
        "mean_latency_s": float(orders["latency_s"].mean()) if len(orders) else float("nan"),
        "median_latency_s": float(orders["latency_s"].median()) if len(orders) else float("nan"),
        "median_valid_order_latency_s": float(valid_orders.loc[valid_orders["completed_before_reset"], "latency_s"].median())
        if len(valid_orders) and valid_orders["completed_before_reset"].any()
        else float("nan"),
        "median_valid_sequence_completion_s": float(valid_sequence_summary["sequence_completion_time_s"].median())
        if len(valid_sequence_summary)
        else float("nan"),
    }


def _load_raw_depth(cfg: ExecutionLatencyGridConfig) -> pd.DataFrame:
    q1_col = q1_column_for_side(raw_side=cfg.raw_side, queue_col=cfg.queue_col)
    columns = ["ts", "order_type", "side", "qty", q1_col]
    raw = pd.read_parquet(cfg.raw_level_path, columns=columns)
    raw = raw.sort_values("ts", kind="mergesort").reset_index(drop=True)
    raw["source_row_pos"] = np.arange(len(raw), dtype=np.int64)
    if raw.empty:
        raise ValueError("raw depth parquet has no rows")
    return raw


def _build_first_level_arrays(
    raw: pd.DataFrame,
    *,
    raw_side: str,
    queue_col: str,
    market_side: str | None = None,
) -> dict[str, Any]:
    q1_col = q1_column_for_side(raw_side=raw_side, queue_col=queue_col)
    consuming_side = market_side_for_queue(
        raw_side=raw_side,
        queue_col=queue_col,
        market_side=market_side,
    )
    ts = raw["ts"]
    origin = ts.iloc[0]
    times = ((ts - origin).dt.total_seconds()).to_numpy(dtype=np.float64)
    q1_post = raw[q1_col].to_numpy(dtype=np.float64)
    q1_pre = np.empty_like(q1_post)
    q1_pre[0] = np.nan
    q1_pre[1:] = q1_post[:-1]
    delta = q1_post - q1_pre

    source_qty = np.rint(raw["qty"].to_numpy(dtype=np.float64)).astype(np.uint32)
    types = raw["order_type"].astype(str).str.lower().to_numpy()
    sides = raw["side"].astype(str).to_numpy()

    dims = np.full(len(raw), IGNORED_DIM, dtype=np.int32)
    qty = np.zeros(len(raw), dtype=np.uint32)
    explained = ~np.isfinite(delta) | (delta == 0.0)
    for typ, dim, sign in (
        ("limit", LIMIT_DIM, 1.0),
        ("cancel", CANCEL_DIM, -1.0),
        ("market", MARKET_DIM, -1.0),
    ):
        event_side = consuming_side if typ == "market" else raw_side
        mask = (
            (sides == event_side)
            & (types == typ)
            & np.isfinite(delta)
            & (sign * delta > 0.0)
        )
        inferred = np.minimum(
            np.rint(np.abs(delta[mask])).astype(np.uint32),
            source_qty[mask],
        )
        positive = inferred > 0
        idx = np.flatnonzero(mask)[positive]
        dims[idx] = dim
        qty[idx] = inferred[positive]
        fully_explained = mask.copy()
        fully_explained[mask] &= np.rint(np.abs(delta[mask])) <= source_qty[mask]
        explained |= fully_explained

    is_queue_reset = np.isfinite(delta) & (delta != 0.0) & ~explained

    return {
        "origin": origin,
        "times": times,
        "ts": ts,
        "source_row_pos": raw["source_row_pos"].to_numpy(dtype=np.int64),
        "q1_post": q1_post,
        "q1_delta": delta,
        "is_queue_reset": is_queue_reset,
        "source_qty": source_qty,
        "dims": dims,
        "qty": qty,
        "q1_col": q1_col,
        "market_side": consuming_side,
    }


def _minute_starts(
    raw: pd.DataFrame,
    cfg: ExecutionLatencyGridConfig,
) -> pd.DatetimeIndex:
    min_ts = raw["ts"].iloc[0]
    max_ts = raw["ts"].iloc[-1]
    start = min_ts.floor(f"{int(cfg.minute_seconds)}s") if cfg.start_time is None else pd.Timestamp(cfg.start_time)
    end = max_ts if cfg.end_time is None else pd.Timestamp(cfg.end_time)
    if getattr(raw["ts"].dt, "tz", None) is not None:
        if start.tzinfo is None:
            start = start.tz_localize(raw["ts"].dt.tz)
        if end.tzinfo is None:
            end = end.tz_localize(raw["ts"].dt.tz)
    last_start = min(end, max_ts - pd.Timedelta(seconds=cfg.n_orders * cfg.order_spacing_seconds))
    return pd.date_range(start=start, end=last_start, freq=pd.Timedelta(seconds=cfg.minute_seconds))


def _window_starts(
    raw: pd.DataFrame,
    arrays: dict[str, Any],
    cfg: ExecutionLatencyGridConfig,
) -> pd.DatetimeIndex:
    policy = cfg.window_start_policy.lower()
    if policy == "clock":
        return _minute_starts(raw, cfg)
    if policy == "first_limit_every":
        return _first_limit_every_starts(raw, arrays, cfg)
    raise ValueError("window_start_policy must be 'clock' or 'first_limit_every'")


def _first_limit_every_starts(
    raw: pd.DataFrame,
    arrays: dict[str, Any],
    cfg: ExecutionLatencyGridConfig,
) -> pd.DatetimeIndex:
    limit_pos = np.flatnonzero((arrays["dims"] == LIMIT_DIM) & (arrays["qty"] > 0))
    if limit_pos.size == 0:
        return pd.DatetimeIndex([])

    ts = raw["ts"]
    keep = np.ones(limit_pos.shape, dtype=bool)
    max_start_s = float(arrays["times"][-1]) - float(cfg.n_orders) * float(
        cfg.order_spacing_seconds
    )
    keep &= arrays["times"][limit_pos] <= max_start_s
    if cfg.start_time is not None:
        start_bound = _timestamp_like(ts, cfg.start_time)
        keep &= (ts.iloc[limit_pos] >= start_bound).to_numpy()
    if cfg.end_time is not None:
        end_bound = _timestamp_like(ts, cfg.end_time)
        keep &= (ts.iloc[limit_pos] <= end_bound).to_numpy()
    limit_pos = limit_pos[keep]
    if limit_pos.size == 0:
        return pd.DatetimeIndex([])

    spacing = float(cfg.minute_seconds)
    if spacing > 0.0:
        limit_seconds = arrays["times"][limit_pos]
        origin = float(limit_seconds[0])
        buckets = np.floor((limit_seconds - origin) / spacing).astype(np.int64)
        _, first_idx = np.unique(buckets, return_index=True)
        limit_pos = limit_pos[np.sort(first_idx)]

    return pd.DatetimeIndex(ts.iloc[limit_pos])


def _timestamp_like(ts: pd.Series, value: object) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if getattr(ts.dt, "tz", None) is not None and out.tzinfo is None:
        out = out.tz_localize(ts.dt.tz)
    return out


def _run_native_latency_grid(
    simproj: Any,
    arrays: dict[str, Any],
    cfg: ExecutionLatencyGridConfig,
    starts: pd.DatetimeIndex,
) -> dict[str, Any]:
    minute_starts_s = np.asarray(
        [(start - arrays["origin"]).total_seconds() for start in starts],
        dtype=np.float64,
    )
    if cfg.tracking_horizon_seconds is None:
        tracking_horizon_seconds = max(float(arrays["times"][-1] - minute_starts_s[0]), 1e-9)
    else:
        tracking_horizon_seconds = float(cfg.tracking_horizon_seconds)

    return simproj.simulate_execution_latency_grid(
        arrays["times"].astype(np.float64, copy=False),
        arrays["dims"].astype(np.int32, copy=False),
        arrays["qty"].astype(np.uint32, copy=False),
        arrays["q1_post"].astype(np.float64, copy=False),
        arrays["source_row_pos"].astype(np.int64, copy=False),
        minute_starts_s,
        int(cfg.n_orders),
        float(cfg.order_spacing_seconds),
        tracking_horizon_seconds,
        cfg.cancellation_policy,
        float(cfg.theta),
        int(cfg.seed),
        bool(cfg.cap_position_by_queue_post),
    )


def _native_latency_frames(
    native: dict[str, Any],
    arrays: dict[str, Any],
    starts: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = _native_orders_frame(native, arrays, starts)
    fills = _native_fills_frame(native, arrays, starts, orders)
    return orders, fills


def _native_orders_frame(
    native: dict[str, Any],
    arrays: dict[str, Any],
    starts: pd.DatetimeIndex,
) -> pd.DataFrame:
    columns = [
        "window_id",
        "window_start",
        "order_slot",
        "order_id",
        "source_row_pos",
        "post_time_s",
        "post_ts",
        "post_qty",
        "source_qty",
        "q1_post",
        "q1_delta",
        "executed_qty",
        "remaining_qty",
        "completed_time_s",
        "completed_ts",
        "latency_s",
        "final_position_qty",
        "final_top_qty",
    ]
    order_ids = np.asarray(native["order_ids"], dtype=np.int64)
    if order_ids.size == 0:
        return pd.DataFrame(columns=columns)

    origin = arrays["origin"]
    window_ids = np.asarray(native["order_window_ids"], dtype=np.int64)
    minute_starts_s = np.asarray(native["order_minute_starts"], dtype=np.float64)
    row_pos = np.asarray(native["order_row_pos"], dtype=np.int64)
    order_times_abs = np.asarray(native["order_times"], dtype=np.float64)
    completed_abs = np.asarray(native["completed_times"], dtype=np.float64)
    completed_mask = np.isfinite(completed_abs)
    completed_time_s = np.where(completed_mask, completed_abs - minute_starts_s, np.nan)

    frame = pd.DataFrame(
        {
            "window_id": window_ids,
            "window_start": [str(starts[int(window_id)]) for window_id in window_ids],
            "order_slot": np.asarray(native["order_slots"], dtype=np.int64),
            "order_id": order_ids,
            "source_row_pos": np.asarray(native["order_source_row_pos"], dtype=np.int64),
            "post_time_s": order_times_abs - minute_starts_s,
            "post_ts": [str(origin + pd.Timedelta(seconds=float(t))) for t in order_times_abs],
            "post_qty": np.asarray(native["initial_qtys"], dtype=np.uint32).astype(np.int64),
            "source_qty": arrays["source_qty"][row_pos].astype(np.int64),
            "q1_post": arrays["q1_post"][row_pos].astype(np.float64),
            "q1_delta": arrays["q1_delta"][row_pos].astype(np.float64),
            "executed_qty": np.asarray(native["executed_qtys"], dtype=np.uint32).astype(np.int64),
            "remaining_qty": np.asarray(native["remaining_qtys"], dtype=np.uint32).astype(np.int64),
            "completed_time_s": completed_time_s,
            "completed_ts": [
                str(origin + pd.Timedelta(seconds=float(t))) if is_completed else ""
                for t, is_completed in zip(completed_abs, completed_mask)
            ],
            "latency_s": np.asarray(native["latencies"], dtype=np.float64),
            "final_position_qty": np.asarray(native["final_position_qtys"], dtype=np.float64),
            "final_top_qty": np.asarray(native["final_top_qtys"], dtype=np.uint32).astype(np.int64),
        }
    )
    return frame[columns]


def _native_fills_frame(
    native: dict[str, Any],
    arrays: dict[str, Any],
    starts: pd.DatetimeIndex,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "window_id",
        "window_start",
        "order_id",
        "order_slot",
        "source_event_row_pos",
        "fill_time_s",
        "fill_ts",
        "qty",
    ]
    fill_order_ids = np.asarray(native["fill_order_ids"], dtype=np.int64)
    if fill_order_ids.size == 0:
        return pd.DataFrame(columns=columns)

    origin = arrays["origin"]
    fill_window_ids = np.asarray(native["fill_window_ids"], dtype=np.int64)
    fill_times_abs = np.asarray(native["fill_times"], dtype=np.float64)
    minute_starts_s = np.asarray(native["minute_starts"], dtype=np.float64)
    order_slot_by_id = orders.set_index("order_id")["order_slot"].to_dict() if not orders.empty else {}

    frame = pd.DataFrame(
        {
            "window_id": fill_window_ids,
            "window_start": [str(starts[int(window_id)]) for window_id in fill_window_ids],
            "order_id": fill_order_ids,
            "order_slot": [order_slot_by_id.get(int(order_id), -1) for order_id in fill_order_ids],
            "source_event_row_pos": np.asarray(native["fill_event_source_row_pos"], dtype=np.int64),
            "fill_time_s": fill_times_abs - minute_starts_s[fill_window_ids],
            "fill_ts": [str(origin + pd.Timedelta(seconds=float(t))) for t in fill_times_abs],
            "qty": np.asarray(native["fill_qtys"], dtype=np.uint32).astype(np.int64),
        }
    )
    return frame[columns]


def _summarize_orders(orders: pd.DataFrame, starts: pd.DatetimeIndex) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "window_id": np.arange(len(starts), dtype=np.int64),
            "window_start": [str(ts) for ts in starts],
        }
    )
    if orders.empty:
        return base.assign(
            n_posted=0,
            n_completed=0,
            completion_rate=np.nan,
            mean_latency_s=np.nan,
            max_latency_s=np.nan,
        )
    grouped = orders.groupby("window_id", sort=True)
    summary = grouped.agg(
        n_posted=("order_id", "count"),
        n_completed=("completed_time_s", lambda x: int(x.notna().sum())),
        mean_latency_s=("latency_s", "mean"),
        median_latency_s=("latency_s", "median"),
        max_latency_s=("latency_s", "max"),
        mean_q1_post=("q1_post", "mean"),
    ).reset_index()
    summary["completion_rate"] = summary["n_completed"] / summary["n_posted"]
    return base.merge(summary, on="window_id", how="left").fillna({"n_posted": 0, "n_completed": 0})


def _sequence_summary(
    orders: pd.DataFrame,
    arrays: dict[str, Any],
    starts: pd.DatetimeIndex,
    cfg: ExecutionLatencyGridConfig,
) -> pd.DataFrame:
    origin = arrays["origin"]
    window_starts_s = np.asarray(
        [(start - origin).total_seconds() for start in starts],
        dtype=np.float64,
    )
    reset_times = arrays["times"][arrays["is_queue_reset"]]
    reset_source_rows = arrays["source_row_pos"][arrays["is_queue_reset"]]
    tracking_horizon = _tracking_horizon_seconds(arrays, cfg, window_starts_s)
    order_groups = {int(key): frame for key, frame in orders.groupby("window_id", sort=False)}

    rows: list[dict[str, Any]] = []
    for window_id, (start_ts, start_s) in enumerate(zip(starts, window_starts_s)):
        end_s = float(start_s) + tracking_horizon
        reset_idx = np.searchsorted(reset_times, start_s, side="left")
        has_reset = reset_idx < len(reset_times) and reset_times[reset_idx] <= end_s
        first_reset_time_s = (
            float(reset_times[reset_idx] - start_s) if has_reset else np.nan
        )
        first_reset_source_row_pos = (
            int(reset_source_rows[reset_idx]) if has_reset else -1
        )
        window_orders = order_groups.get(window_id)
        n_posted = 0 if window_orders is None else int(len(window_orders))
        full_sequence = n_posted == int(cfg.n_orders)
        if window_orders is None or window_orders.empty:
            n_completed_before_reset = 0
            sequence_completion_time_s = np.nan
        else:
            completed = window_orders["completed_time_s"].notna()
            if np.isfinite(first_reset_time_s):
                completed &= window_orders["completed_time_s"] <= first_reset_time_s
            n_completed_before_reset = int(completed.sum())
            if n_completed_before_reset == n_posted and n_posted > 0:
                sequence_completion_time_s = float(
                    window_orders.loc[completed, "completed_time_s"].max()
                )
            else:
                sequence_completion_time_s = np.nan

        sequence_completed_before_reset = (
            n_posted > 0 and n_completed_before_reset == n_posted
        )
        valid_sequence = n_posted > 0
        if cfg.require_full_sequence:
            valid_sequence &= full_sequence
        if cfg.require_sequence_completion:
            valid_sequence &= sequence_completed_before_reset
        if cfg.exclude_queue_reset_windows:
            if cfg.require_sequence_completion:
                valid_sequence &= sequence_completed_before_reset
            else:
                valid_sequence &= not np.isfinite(first_reset_time_s)

        rows.append(
            {
                "window_id": int(window_id),
                "window_start": str(start_ts),
                "n_posted": n_posted,
                "full_sequence": bool(full_sequence),
                "n_completed_before_reset": n_completed_before_reset,
                "sequence_completed_before_reset": bool(sequence_completed_before_reset),
                "sequence_completion_time_s": sequence_completion_time_s,
                "first_reset_time_s": first_reset_time_s,
                "first_reset_source_row_pos": first_reset_source_row_pos,
                "valid_sequence": bool(valid_sequence),
            }
        )
    return pd.DataFrame(rows)


def _annotate_orders_with_sequence_status(
    orders: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    cfg: ExecutionLatencyGridConfig,
) -> pd.DataFrame:
    if orders.empty:
        return orders.assign(
            first_reset_time_s=pd.Series(dtype=np.float64),
            full_sequence=pd.Series(dtype=bool),
            sequence_completed_before_reset=pd.Series(dtype=bool),
            valid_sequence=pd.Series(dtype=bool),
            completed_before_reset=pd.Series(dtype=bool),
            valid_completed_order=pd.Series(dtype=bool),
        )
    cols = [
        "window_id",
        "first_reset_time_s",
        "full_sequence",
        "sequence_completed_before_reset",
        "valid_sequence",
    ]
    out = orders.merge(sequence_summary[cols], on="window_id", how="left")
    completed = out["completed_time_s"].notna()
    reset = out["first_reset_time_s"].to_numpy(dtype=np.float64)
    completed_time = out["completed_time_s"].to_numpy(dtype=np.float64)
    before_reset = completed.to_numpy(dtype=bool) & (
        ~np.isfinite(reset) | (completed_time <= reset)
    )
    out["completed_before_reset"] = before_reset
    out["valid_completed_order"] = out["valid_sequence"].to_numpy(dtype=bool) & before_reset
    if cfg.require_full_sequence:
        out["valid_completed_order"] &= out["full_sequence"].to_numpy(dtype=bool)
    return out


def _annotate_fills_with_sequence_status(
    fills: pd.DataFrame,
    sequence_summary: pd.DataFrame,
) -> pd.DataFrame:
    if fills.empty:
        return fills.assign(
            first_reset_time_s=pd.Series(dtype=np.float64),
            valid_sequence=pd.Series(dtype=bool),
            fill_before_reset=pd.Series(dtype=bool),
        )
    cols = ["window_id", "first_reset_time_s", "valid_sequence"]
    out = fills.merge(sequence_summary[cols], on="window_id", how="left")
    reset = out["first_reset_time_s"].to_numpy(dtype=np.float64)
    fill_time = out["fill_time_s"].to_numpy(dtype=np.float64)
    out["fill_before_reset"] = ~np.isfinite(reset) | (fill_time <= reset)
    return out


def _valid_orders(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return orders.copy()
    return orders[orders["valid_sequence"]].copy()


def _valid_fills(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return fills.copy()
    return fills[fills["valid_sequence"] & fills["fill_before_reset"]].copy()


def _summarize_valid_order_latencies(valid_orders: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "slot_label",
        "order_slot",
        "n_completed",
        "median_latency_s",
        "mean_latency_s",
        "q25_latency_s",
        "q75_latency_s",
        "max_latency_s",
    ]
    if valid_orders.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})

    completed = valid_orders[valid_orders["completed_before_reset"]].copy()
    if completed.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})

    def summarize(frame: pd.DataFrame, slot_label: str, order_slot: int) -> dict[str, Any]:
        values = frame["latency_s"].dropna().to_numpy(dtype=np.float64)
        return {
            "slot_label": slot_label,
            "order_slot": int(order_slot),
            "n_completed": int(values.size),
            "median_latency_s": float(np.median(values)) if values.size else np.nan,
            "mean_latency_s": float(np.mean(values)) if values.size else np.nan,
            "q25_latency_s": float(np.quantile(values, 0.25)) if values.size else np.nan,
            "q75_latency_s": float(np.quantile(values, 0.75)) if values.size else np.nan,
            "max_latency_s": float(np.max(values)) if values.size else np.nan,
        }

    rows = [summarize(completed, "all", -1)]
    for slot, frame in completed.groupby("order_slot", sort=True):
        rows.append(summarize(frame, f"slot_{int(slot)}", int(slot)))
    return pd.DataFrame(rows, columns=columns)


def _tracking_horizon_seconds(
    arrays: dict[str, Any],
    cfg: ExecutionLatencyGridConfig,
    minute_starts_s: np.ndarray,
) -> float:
    if cfg.tracking_horizon_seconds is None:
        return max(float(arrays["times"][-1] - minute_starts_s[0]), 1e-9)
    return float(cfg.tracking_horizon_seconds)


def _plot_latency_by_minute(path: Path, orders: pd.DataFrame, cfg: ExecutionLatencyGridConfig) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5.8))
    if not orders.empty:
        for slot in range(cfg.n_orders):
            slot_rows = orders[orders["order_slot"] == slot]
            ax.scatter(
                slot_rows["window_id"],
                slot_rows["latency_s"],
                s=13,
                alpha=0.72,
                label=f"order {slot + 1}",
            )
    ax.set_title("Passive execution latency by minute window")
    ax.set_xlabel("minute window id")
    ax.set_ylabel("latency to full execution (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_latency_histogram(path: Path, orders: pd.DataFrame) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.4))
    latencies = orders["latency_s"].dropna().to_numpy(dtype=np.float64) if not orders.empty else np.array([])
    if latencies.size > 0:
        ax.hist(latencies, bins=40, color="#557aa6", alpha=0.82)
        ax.axvline(np.mean(latencies), color="#b83232", linewidth=1.5, label=f"mean {np.mean(latencies):.1f}s")
        ax.axvline(np.median(latencies), color="#1f7a3d", linewidth=1.5, label=f"median {np.median(latencies):.1f}s")
    ax.set_title("Passive execution latency distribution")
    ax.set_xlabel("latency to full execution (s)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _setup_matplotlib() -> None:
    mpl_cache = Path("/private/tmp/matplotlib-cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")


def _import_simproj():
    try:
        import simproj  # type: ignore

        return simproj
    except ModuleNotFoundError:
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        code_python = repo_root / "code" / "python"
        if str(code_python) not in sys.path:
            sys.path.insert(0, str(code_python))
        import simproj  # type: ignore

        return simproj


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-level-path", default=ExecutionLatencyGridConfig.raw_level_path)
    parser.add_argument("--output-dir", default=ExecutionLatencyGridConfig.output_dir)
    parser.add_argument("--start-time", default=ExecutionLatencyGridConfig.start_time)
    parser.add_argument("--end-time", default=ExecutionLatencyGridConfig.end_time)
    parser.add_argument(
        "--window-start-policy",
        choices=["clock", "first_limit_every"],
        default=ExecutionLatencyGridConfig.window_start_policy,
    )
    parser.add_argument("--minute-seconds", type=float, default=ExecutionLatencyGridConfig.minute_seconds)
    parser.add_argument("--n-orders", type=int, default=ExecutionLatencyGridConfig.n_orders)
    parser.add_argument("--order-spacing-seconds", type=float, default=ExecutionLatencyGridConfig.order_spacing_seconds)
    parser.add_argument("--tracking-horizon-seconds", type=float, default=ExecutionLatencyGridConfig.tracking_horizon_seconds)
    parser.add_argument("--raw-side", default=ExecutionLatencyGridConfig.raw_side)
    parser.add_argument("--queue-col", default=ExecutionLatencyGridConfig.queue_col)
    parser.add_argument("--market-side", default=ExecutionLatencyGridConfig.market_side)
    parser.add_argument("--cancellation-policy", default=ExecutionLatencyGridConfig.cancellation_policy)
    parser.add_argument("--theta", type=float, default=ExecutionLatencyGridConfig.theta)
    parser.add_argument("--cap-position-by-queue-post", action="store_true")
    parser.add_argument("--require-full-sequence", action="store_true")
    parser.add_argument("--require-sequence-completion", action="store_true")
    parser.add_argument("--exclude-queue-reset-windows", action="store_true")
    parser.add_argument("--seed", type=int, default=ExecutionLatencyGridConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ExecutionLatencyGridConfig(**vars(args))
    summary = run_execution_latency_grid(cfg)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
