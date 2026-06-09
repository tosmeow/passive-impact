"""First-queue execution utilities for the impact-cost experiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def opposite_side(side: str) -> str:
    """Return the opposite raw side label for the two-sided data convention."""
    side = str(side).upper()
    if side == "A":
        return "B"
    if side == "B":
        return "A"
    raise ValueError("side must be 'A' or 'B'")


def market_side_for_queue(
    *,
    raw_side: str,
    queue_col: str | None = None,
    market_side: str | None = None,
) -> str:
    """Return the market side that consumes the selected queue.

    In the available depth data, passive bid-side limit/cancel events are
    labelled `A` and consume `q_b`/`b_1` through `market B`; passive ask-side
    limit/cancel events are labelled `B` and consume `q_a`/`a_1` through
    `market A`.
    """
    if market_side is not None:
        out = str(market_side).upper()
        if out not in {"A", "B"}:
            raise ValueError("market_side must be 'A' or 'B'")
        return out

    if queue_col in {"q_a", "a_1"}:
        return "A"
    if queue_col in {"q_b", "b_1"}:
        return "B"
    return opposite_side(raw_side)


def price_sign_for_queue(queue_col: str) -> float:
    """Return the price-sign multiplier for a single-queue contribution.

    The single-queue impact primitives return a queue-displacement contribution
    for the chosen queue. Ask-side contributions enter price with `+`;
    bid-side contributions enter with `-`.
    """
    if queue_col in {"q_b", "b_1"}:
        return -1.0
    if queue_col in {"q_a", "a_1"}:
        return 1.0
    raise ValueError("queue_col must identify the ask or bid queue")


def q1_column_for_side(*, raw_side: str, queue_col: str | None = None) -> str:
    """Return the first-depth column used for execution on the posting side.

    The aggregated experiment pairs B-side posting with `q_a` and A-side
    posting with `q_b`; the raw depth file carries those as `a_1` and `b_1`.
    """
    if queue_col == "q_a":
        return "a_1"
    if queue_col == "q_b":
        return "b_1"
    side = raw_side.upper()
    if side == "B":
        return "a_1"
    if side == "A":
        return "b_1"
    raise ValueError("raw_side must be 'A' or 'B'")


def first_level_execution_events_from_snapshots(
    df: pd.DataFrame,
    *,
    raw_side: str,
    q1_col: str,
    market_side: str | None = None,
    qty_col: str = "qty",
    ts_col: str = "ts",
    order_type_col: str = "order_type",
    side_col: str = "side",
    previous_q1: float | None = None,
) -> pd.DataFrame:
    """Infer level-1 L/C/N event sizes from post-event q1 snapshots.

    The output keeps one row per input row. Rows that do not affect q1 on
    `raw_side` get `level = 0` and `qty = 0`; rows that do affect q1 get
    `level = 1` and `qty` equal to the inferred first-queue size.
    """
    required = {ts_col, order_type_col, side_col, qty_col, q1_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"missing columns for first-level execution: {sorted(missing)}")

    q1_post = df[q1_col].to_numpy(dtype=np.float64)
    q1_pre = np.empty_like(q1_post)
    if len(q1_post) > 0:
        q1_pre[0] = np.nan if previous_q1 is None else float(previous_q1)
        q1_pre[1:] = q1_post[:-1]

    delta = q1_post - q1_pre
    source_qty = np.rint(df[qty_col].to_numpy(dtype=np.float64)).astype(np.int64)
    if np.any(source_qty < 0):
        raise ValueError("qty must be non-negative")

    types = df[order_type_col].astype(str).str.lower().to_numpy()
    sides = df[side_col].astype(str).to_numpy()
    posting_side = str(raw_side).upper()
    consuming_side = market_side_for_queue(
        raw_side=raw_side,
        queue_col=q1_col,
        market_side=market_side,
    )

    inferred_qty = np.zeros(len(df), dtype=np.int64)
    level = np.zeros(len(df), dtype=np.int64)
    for typ, sign in (("limit", 1.0), ("cancel", -1.0), ("market", -1.0)):
        event_side = consuming_side if typ == "market" else posting_side
        mask = (
            (sides == event_side)
            & (types == typ)
            & np.isfinite(delta)
            & (sign * delta > 0.0)
        )
        qty = np.minimum(np.rint(np.abs(delta[mask])).astype(np.int64), source_qty[mask])
        positive = qty > 0
        mask_pos = np.flatnonzero(mask)[positive]
        inferred_qty[mask_pos] = qty[positive]
        level[mask_pos] = 1

    out = pd.DataFrame(
        {
            "ts": df[ts_col].to_numpy(),
            "order_type": types,
            "side": sides,
            "qty": inferred_qty,
            "source_qty": source_qty,
            "q1": q1_post,
            "q1_pre": q1_pre,
            "q1_delta": delta,
            "level": level,
        }
    )
    if "source_row_pos" in df.columns:
        out["source_row_pos"] = df["source_row_pos"].to_numpy(dtype=np.int64)
    else:
        out["source_row_pos"] = np.arange(len(df), dtype=np.int64)
    return out


def load_first_level_execution_window(
    parquet_path: str | Path,
    *,
    start_time: object,
    horizon_seconds: float,
    raw_side: str,
    queue_col: str | None = None,
    market_side: str | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp, str]:
    """Load a time window from raw depth data and infer first-level events."""
    q1_col = q1_column_for_side(raw_side=raw_side, queue_col=queue_col)
    columns = ["ts", "order_type", "side", "qty", q1_col]
    raw = pd.read_parquet(parquet_path, columns=columns)
    raw = raw.sort_values("ts", kind="mergesort").reset_index(drop=True)
    raw["source_row_pos"] = np.arange(len(raw), dtype=np.int64)
    if raw.empty:
        raise ValueError("raw depth parquet has no rows")

    start = pd.Timestamp(start_time)
    if getattr(raw["ts"].dt, "tz", None) is not None and start.tzinfo is None:
        start = start.tz_localize(raw["ts"].dt.tz)
    end = start + pd.Timedelta(seconds=horizon_seconds)

    mask = ((raw["ts"] >= start) & (raw["ts"] <= end)).to_numpy()
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        raise ValueError(f"no rows found in [{start}, {end}]")

    lo = max(0, int(positions[0]) - 1)
    hi = int(positions[-1]) + 1
    with_previous = raw.iloc[lo:hi].copy().reset_index(drop=True)
    previous_q1 = None if lo == positions[0] else float(with_previous[q1_col].iloc[0])
    window = with_previous[with_previous["ts"] >= start].copy().reset_index(drop=True)
    execution = first_level_execution_events_from_snapshots(
        window,
        raw_side=raw_side,
        q1_col=q1_col,
        market_side=market_side,
        previous_q1=previous_q1,
    )
    return execution, start, q1_col
