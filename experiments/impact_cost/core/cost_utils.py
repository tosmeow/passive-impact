"""Small utilities for passive execution-cost experiments.

This module is intentionally experiment-local.  It adapts real-data rows with
sizes to the library's current unit-event representation, flags passive limit
orders, and tracks when flagged passive orders would be executed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


LIMIT = "limit"
CANCEL = "cancel"
MARKET = "market"


@dataclass(frozen=True)
class TrackingResult:
    """Passive execution tracker output.

    `orders` has one row per flagged passive order. `fills` has one row per
    market event that executes part of an order. `ledger` is optional and, when
    requested, records the per-event position and remaining-quantity
    transitions.
    """

    orders: pd.DataFrame
    fills: pd.DataFrame
    ledger: Optional[pd.DataFrame] = None


@dataclass
class _ActiveOrder:
    order_id: int
    row_pos: int
    l_index: int
    time: float
    initial_qty: int
    remaining_qty: int
    position_qty: float
    top_qty: int = 0
    completed_time: Optional[float] = None


def _normalised_types(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].astype(str).str.lower().to_numpy()


def _normalised_sides(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].astype(str).to_numpy()


def event_seconds(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    origin: object | None = None,
) -> np.ndarray:
    """Return event timestamps in seconds from `origin`.

    If `origin` is omitted, the first timestamp in `df` is used. Numeric
    timestamps are treated as already being in seconds.
    """
    ts = df[ts_col]
    if is_numeric_dtype(ts):
        values = ts.to_numpy(dtype=np.float64)
        base = values[0] if origin is None else float(origin)
        return values - base

    dt = pd.to_datetime(ts)
    base = dt.iloc[0] if origin is None else pd.Timestamp(origin)
    return ((dt - base).dt.total_seconds()).to_numpy(dtype=np.float64)


def limit_event_positions(
    df: pd.DataFrame,
    *,
    side: str | None = None,
    level_col: str | None = None,
    target_level: object = 1,
    order_type_col: str = "order_type",
    side_col: str = "side",
) -> np.ndarray:
    """Return row positions of limit events, optionally filtered by side."""
    types = _normalised_types(df, order_type_col)
    mask = types == LIMIT
    if side is not None:
        mask &= _normalised_sides(df, side_col) == side
    mask &= _level_mask(df, level_col=level_col, target_level=target_level)
    return np.flatnonzero(mask)


def flag_passive_limits(
    df: pd.DataFrame,
    policy: str,
    *,
    side: str | None = None,
    level_col: str | None = None,
    target_level: object = 1,
    every_seconds: float | None = None,
    fraction: float | None = None,
    indices: Iterable[int] | None = None,
    index_base: int = 1,
    seed: int | None = None,
    ts_col: str = "ts",
    order_type_col: str = "order_type",
    side_col: str = "side",
) -> np.ndarray:
    """Flag passive limit-order rows according to a simple policy.

    Policies:
    - `first_every`: first matching L event in each `every_seconds` time bucket.
    - `random_fraction`: exactly `round(fraction * n_L)` matching L events.
    - `indices`: explicit positions within matching L events. With the default
      `index_base=1`, index `5` means "the fifth matching L event".

    When `level_col` is provided, policies are applied only to rows with
    `level_col == target_level`, matching the first-queue execution convention
    before any queue-level aggregation.
    """
    l_pos = limit_event_positions(
        df,
        side=side,
        level_col=level_col,
        target_level=target_level,
        order_type_col=order_type_col,
        side_col=side_col,
    )
    flags = np.zeros(len(df), dtype=bool)
    if len(l_pos) == 0:
        return flags

    policy = policy.lower()
    if policy == "first_every":
        if every_seconds is None or every_seconds <= 0:
            raise ValueError("every_seconds must be positive for first_every")
        seconds = event_seconds(df, ts_col=ts_col)[l_pos]
        origin = seconds[0]
        buckets = np.floor((seconds - origin) / every_seconds).astype(np.int64)
        _, first_idx = np.unique(buckets, return_index=True)
        flags[l_pos[np.sort(first_idx)]] = True
        return flags

    if policy == "random_fraction":
        if fraction is None or not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1] for random_fraction")
        n_selected = int(round(fraction * len(l_pos)))
        if n_selected == 0:
            return flags
        rng = np.random.default_rng(seed)
        chosen = rng.choice(l_pos, size=n_selected, replace=False)
        flags[np.sort(chosen)] = True
        return flags

    if policy == "indices":
        if indices is None:
            raise ValueError("indices must be provided for indices policy")
        selected = []
        for raw_idx in indices:
            idx = int(raw_idx) - index_base
            if idx < 0 or idx >= len(l_pos):
                raise IndexError(
                    f"limit index {raw_idx} is outside 1..{len(l_pos)}"
                    if index_base == 1
                    else f"limit index {raw_idx} is outside valid range"
                )
            selected.append(l_pos[idx])
        flags[selected] = True
        return flags

    raise ValueError(
        "policy must be one of 'first_every', 'random_fraction', or 'indices'"
    )


def expand_event_times_by_dim(
    df: pd.DataFrame,
    dim_map: Mapping[tuple[str, str], int],
    *,
    qty_col: str = "qty",
    ts_col: str = "ts",
    order_type_col: str = "order_type",
    side_col: str = "side",
    origin: object | None = None,
) -> dict[int, np.ndarray]:
    """Expand sized rows into repeated unit-event times grouped by model dim.

    `dim_map` deliberately keeps the raw side convention outside the library.
    Example: `{("limit", "B"): 0, ("cancel", "B"): 1, ("market", "B"): 2}`.

    Quantity convention: a row with `qty=n` becomes `n` unit events at the same
    timestamp. When used for conditional simulation, these units are processed
    sequentially by the existing simulator, so factual and counterfactual
    intensities are recomputed after each unit. This differs slightly from a
    frozen-pre-event convention that would test all `n` uniforms against
    lambda(q_t-) and lambda(q_bar_t-). See `experiments/impact_cost/README.md`.
    """
    seconds = event_seconds(df, ts_col=ts_col, origin=origin)
    types = _normalised_types(df, order_type_col)
    sides = _normalised_sides(df, side_col)
    qty = df[qty_col].to_numpy(dtype=np.int64)
    if np.any(qty < 0):
        raise ValueError("qty must be non-negative")

    out: dict[int, list[np.ndarray]] = {}
    for (order_type, side), dim in dim_map.items():
        mask = (types == order_type.lower()) & (sides == side)
        repeated = np.repeat(seconds[mask], qty[mask]).astype(np.float64)
        out.setdefault(int(dim), []).append(repeated)

    return {
        dim: np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        for dim, parts in out.items()
    }


def regroup_event_times_by_dim(
    events_by_dim: Mapping[int, Iterable[float]] | None = None,
    *,
    times: Iterable[float] | None = None,
    dims: Iterable[int] | None = None,
    time_decimals: int | None = None,
) -> pd.DataFrame:
    """Regroup flattened unit events into sized `(time, dim, qty)` rows.

    This is the companion to `expand_event_times_by_dim`. After conditional
    simulation on flattened unit events, pass either:

    - a mapping `{dim: repeated_times}`, or
    - parallel arrays `times=sim_result.times(), dims=sim_result.dims()`.

    The result has columns `time`, `dim`, and `qty`, sorted by time then dim.
    By default grouping uses exact floating-point equality, which is appropriate
    for repeated timestamps produced by expansion. Use `time_decimals` only when
    you intentionally want to coalesce tiny numerical timestamp differences.
    """
    if events_by_dim is not None and (times is not None or dims is not None):
        raise ValueError("provide either events_by_dim or times/dims, not both")

    if events_by_dim is not None:
        pieces = []
        for dim, dim_times in events_by_dim.items():
            arr = np.asarray(list(dim_times), dtype=np.float64)
            if arr.size == 0:
                continue
            pieces.append(
                pd.DataFrame(
                    {
                        "time": arr,
                        "dim": np.full(arr.shape, int(dim), dtype=np.int64),
                    }
                )
            )
        if not pieces:
            return pd.DataFrame(
                {
                    "time": pd.Series(dtype=np.float64),
                    "dim": pd.Series(dtype=np.int64),
                    "qty": pd.Series(dtype=np.int64),
                }
            )
        flat = pd.concat(pieces, ignore_index=True)
    else:
        if times is None or dims is None:
            raise ValueError("times and dims must be provided together")
        time_arr = np.asarray(list(times), dtype=np.float64)
        dim_arr = np.asarray(list(dims), dtype=np.int64)
        if time_arr.shape != dim_arr.shape:
            raise ValueError("times and dims must have the same shape")
        flat = pd.DataFrame({"time": time_arr, "dim": dim_arr})

    if time_decimals is not None:
        flat["time"] = flat["time"].round(int(time_decimals))

    grouped = (
        flat.groupby(["time", "dim"], sort=True)
        .size()
        .rename("qty")
        .reset_index()
    )
    grouped["qty"] = grouped["qty"].astype(np.int64)
    grouped["dim"] = grouped["dim"].astype(np.int64)
    return grouped[["time", "dim", "qty"]]


def _market_fill_qty(position: float, remaining: int, event_qty: int) -> int:
    if remaining <= 0 or event_qty <= 0:
        return 0
    ahead_before = max(position - remaining, 0.0)
    consumed_until = min(position, float(event_qty))
    return int(max(0.0, consumed_until - ahead_before))


def track_passive_fills(
    df: pd.DataFrame,
    passive_l_flags: Iterable[bool],
    *,
    side: str | None = None,
    market_side: str | None = None,
    queue_col: str | None = None,
    level_col: str | None = None,
    target_level: object = 1,
    cancellation_policy: str = "top",
    theta: float = 1.0,
    seed: int | None = None,
    qty_col: str = "qty",
    ts_col: str = "ts",
    order_type_col: str = "order_type",
    side_col: str = "side",
    include_ledger: bool = False,
) -> TrackingResult:
    """Track execution of flagged passive limit orders.

    The scalar position convention is:
    - a flagged L starts with position equal to the post-event queue snapshot
      in `queue_col` when provided, otherwise to its own `qty`. With
      `level_col`, this should be the first-queue post-event snapshot;
    - later first-queue L events on the same selected side are treated as being
      on top of the order and build a top-side buffer, but do not change
      position;
    - first-queue N events consume queue volume from the top and fill the order
      when they overlap the order's position interval;
    - first-queue C events follow `cancellation_policy`.

    For `cancellation_policy='top'`, cancellations first consume the top-side
    buffer and do not change position while that buffer is positive. If a top
    cancellation asks to remove more volume than is available on top, the
    residual decreases position. This handles the edge case where our order is
    last in queue and no top-side volume exists yet. For `probabilistic_top`,
    each cancellation unit attempts to be top-side with probability `theta`;
    impossible top cancellations fall through to the position-decreasing part.
    """
    flags = np.asarray(list(passive_l_flags), dtype=bool)
    if flags.shape != (len(df),):
        raise ValueError("passive_l_flags must have one boolean per dataframe row")

    active_level = _level_mask(df, level_col=level_col, target_level=target_level)
    bad_flags = np.flatnonzero(flags & ~active_level)
    if bad_flags.size > 0:
        raise ValueError(
            "passive_l_flags may only select rows on the execution target level; "
            f"first invalid row_pos={int(bad_flags[0])}"
        )

    seconds = event_seconds(df, ts_col=ts_col)
    types = _normalised_types(df, order_type_col)
    sides = _normalised_sides(df, side_col)
    qty = df[qty_col].to_numpy(dtype=np.int64)
    queue = None if queue_col is None else df[queue_col].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    posting_side = None if side is None else str(side).upper()
    consuming_side = posting_side if market_side is None else str(market_side).upper()

    l_positions = limit_event_positions(
        df,
        side=posting_side,
        level_col=level_col,
        target_level=target_level,
        order_type_col=order_type_col,
        side_col=side_col,
    )
    l_rank_by_row = {int(row_pos): rank + 1 for rank, row_pos in enumerate(l_positions)}

    orders: list[_ActiveOrder] = []
    fills: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    next_order_id = 0
    global_top_qty = 0

    for row_pos in range(len(df)):
        if not active_level[row_pos]:
            continue

        typ = types[row_pos]
        if posting_side is not None:
            event_side = consuming_side if typ == MARKET else posting_side
            if sides[row_pos] != event_side:
                continue
        event_qty = int(qty[row_pos])
        if event_qty <= 0 or typ not in {LIMIT, CANCEL, MARKET}:
            continue

        time = float(seconds[row_pos])

        if typ == LIMIT:
            is_own_limit = bool(flags[row_pos])
            if not is_own_limit:
                active_orders = [order for order in orders if order.remaining_qty > 0]
                for order in active_orders:
                    before_pos = order.position_qty
                    before_top = global_top_qty
                    order.top_qty = global_top_qty + event_qty
                    if include_ledger:
                        ledger.append(
                            _ledger_row(
                                row_pos, time, typ, event_qty, order,
                                before_pos, order.position_qty,
                                order.remaining_qty, order.remaining_qty,
                                top_before=before_top, top_after=global_top_qty + event_qty,
                                fill_qty=0,
                                cancel_position_qty=0,
                                cancel_top_qty=0,
                            )
                        )
                if active_orders:
                    global_top_qty += event_qty
                    for order in active_orders:
                        order.top_qty = global_top_qty

            if is_own_limit:
                initial_qty = event_qty
                initial_position = float(queue[row_pos]) if queue is not None else initial_qty
                order = _ActiveOrder(
                    order_id=next_order_id,
                    row_pos=row_pos,
                    l_index=l_rank_by_row.get(row_pos, -1),
                    time=time,
                    initial_qty=initial_qty,
                    remaining_qty=initial_qty,
                    position_qty=max(
                        initial_position,
                        _minimum_new_order_position(orders, initial_qty),
                    ),
                    top_qty=global_top_qty,
                )
                next_order_id += 1
                orders.append(order)
                _enforce_order_positions(orders)
                if include_ledger:
                    ledger.append(
                        _ledger_row(
                            row_pos, time, "own_limit", event_qty, order,
                            np.nan, order.position_qty,
                            0, order.remaining_qty,
                            top_before=global_top_qty, top_after=global_top_qty,
                            fill_qty=0,
                            cancel_position_qty=0,
                            cancel_top_qty=0,
                        )
                    )
            continue

        if typ == MARKET:
            for order in orders:
                if order.remaining_qty <= 0:
                    continue
                before_pos = order.position_qty
                before_remaining = order.remaining_qty
                before_top = global_top_qty
                fill_qty = _market_fill_qty(before_pos, before_remaining, event_qty)
                order.remaining_qty -= fill_qty
                order.position_qty = (
                    0.0
                    if order.remaining_qty == 0
                    else max(float(order.remaining_qty), before_pos - event_qty)
                )
                order.top_qty = global_top_qty
                if order.remaining_qty == 0 and order.completed_time is None:
                    order.completed_time = time
                if fill_qty > 0:
                    fills.append(
                        {
                            "order_id": order.order_id,
                            "order_row_pos": order.row_pos,
                            "l_index": order.l_index,
                            "event_row_pos": row_pos,
                            "time": time,
                            "qty": fill_qty,
                        }
                    )
                if include_ledger:
                    ledger.append(
                        _ledger_row(
                            row_pos, time, typ, event_qty, order,
                            before_pos, order.position_qty,
                            before_remaining, order.remaining_qty,
                            top_before=before_top, top_after=global_top_qty,
                            fill_qty=fill_qty,
                            cancel_position_qty=0,
                            cancel_top_qty=0,
                        )
                    )
            _enforce_order_positions(orders)
            continue

        if typ == CANCEL:
            top_before_event = global_top_qty
            desired_top_qty = _desired_top_cancel_qty(
                event_qty, policy=cancellation_policy, theta=theta, rng=rng
            )
            cancel_top_qty = min(global_top_qty, desired_top_qty)
            global_top_qty -= cancel_top_qty
            cancel_position_qty = event_qty - cancel_top_qty
            for order in orders:
                if order.remaining_qty <= 0:
                    continue
                before_pos = order.position_qty
                before_remaining = order.remaining_qty
                before_top = top_before_event
                order.position_qty = max(
                    float(order.remaining_qty),
                    order.position_qty - cancel_position_qty,
                )
                order.top_qty = global_top_qty
                if include_ledger:
                    ledger.append(
                        _ledger_row(
                            row_pos, time, typ, event_qty, order,
                            before_pos, order.position_qty,
                            before_remaining, order.remaining_qty,
                            top_before=before_top, top_after=global_top_qty,
                            fill_qty=0,
                            cancel_position_qty=cancel_position_qty,
                            cancel_top_qty=cancel_top_qty,
                        )
                    )
            _enforce_order_positions(orders)

    orders_df = pd.DataFrame(
        [
            {
                "order_id": order.order_id,
                "row_pos": order.row_pos,
                "l_index": order.l_index,
                "time": order.time,
                "initial_qty": order.initial_qty,
                "executed_qty": order.initial_qty - order.remaining_qty,
                "remaining_qty": order.remaining_qty,
                "final_position_qty": order.position_qty,
                "final_top_qty": order.top_qty,
                "completed_time": order.completed_time,
            }
            for order in orders
        ]
    )
    fills_df = pd.DataFrame(
        fills,
        columns=[
            "order_id",
            "order_row_pos",
            "l_index",
            "event_row_pos",
            "time",
            "qty",
        ],
    )
    ledger_df = None if not include_ledger else pd.DataFrame(ledger)
    return TrackingResult(orders=orders_df, fills=fills_df, ledger=ledger_df)


def _level_mask(
    df: pd.DataFrame,
    *,
    level_col: str | None,
    target_level: object,
) -> np.ndarray:
    if level_col is None:
        return np.ones(len(df), dtype=bool)
    if level_col not in df.columns:
        raise KeyError(f"level column {level_col!r} is not present")
    return (df[level_col] == target_level).to_numpy(dtype=bool)


def _minimum_new_order_position(orders: list[_ActiveOrder], qty: int) -> float:
    active_positions = [
        order.position_qty
        for order in orders
        if order.remaining_qty > 0
    ]
    if not active_positions:
        return float(qty)
    return max(active_positions) + float(qty)


def _enforce_order_positions(orders: list[_ActiveOrder]) -> None:
    """Keep active own orders in posting order without crossing."""
    min_position = 0.0
    for order in orders:
        if order.remaining_qty <= 0:
            order.position_qty = 0.0
            continue
        min_position += float(order.remaining_qty)
        order.position_qty = max(order.position_qty, min_position)
        min_position = order.position_qty


def _ledger_row(
    row_pos: int,
    time: float,
    event_type: str,
    event_qty: int,
    order: _ActiveOrder,
    position_before: float,
    position_after: float,
    remaining_before: int,
    remaining_after: int,
    *,
    top_before: int,
    top_after: int,
    fill_qty: int,
    cancel_position_qty: int,
    cancel_top_qty: int,
) -> dict[str, object]:
    return {
        "event_row_pos": row_pos,
        "time": time,
        "event_type": event_type,
        "event_qty": event_qty,
        "order_id": order.order_id,
        "order_row_pos": order.row_pos,
        "l_index": order.l_index,
        "position_before": position_before,
        "position_after": position_after,
        "remaining_before": remaining_before,
        "remaining_after": remaining_after,
        "top_before": top_before,
        "top_after": top_after,
        "fill_qty": fill_qty,
        "cancel_top_qty": cancel_top_qty,
        "cancel_position_qty": cancel_position_qty,
    }


def _desired_top_cancel_qty(
    event_qty: int,
    *,
    policy: str,
    theta: float,
    rng: np.random.Generator,
) -> int:
    policy = policy.lower()
    if policy == "top":
        return int(event_qty)
    if policy in {"position", "below"}:
        return 0
    if policy in {"probabilistic_top", "top_probability"}:
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")
        return int(rng.binomial(int(event_qty), theta))
    raise ValueError(
        "cancellation_policy must be 'top', 'position', 'below', "
        "or 'probabilistic_top'"
    )


def cost_from_fills(
    fills: pd.DataFrame,
    impact_times: np.ndarray,
    impact_values: np.ndarray,
    *,
    time_col: str = "time",
    qty_col: str = "qty",
) -> np.ndarray:
    """Compute passive cost from fill times and an impact path.

    Uses the left-limit value at each fill time, matching the `P_{t-}` convention.
    `impact_values` can be shape `(n_times,)` or `(n_times, n_paths)`.
    """
    values = np.asarray(impact_values, dtype=np.float64)
    if values.ndim == 1:
        values_2d = values[:, None]
    elif values.ndim == 2:
        values_2d = values
    else:
        raise ValueError("impact_values must be one- or two-dimensional")

    times = np.asarray(impact_times, dtype=np.float64)
    if len(times) != values_2d.shape[0]:
        raise ValueError("impact_times length must match impact_values rows")

    if fills.empty:
        out = np.zeros(values_2d.shape[1], dtype=np.float64)
        return out[0] if values.ndim == 1 else out

    fill_times = fills[time_col].to_numpy(dtype=np.float64)
    fill_qty = fills[qty_col].to_numpy(dtype=np.float64)
    idx = np.searchsorted(times, fill_times, side="left") - 1

    contributions = np.zeros((len(fill_times), values_2d.shape[1]), dtype=np.float64)
    valid = idx >= 0
    contributions[valid] = values_2d[idx[valid]] * fill_qty[valid, None]
    out = contributions.sum(axis=0)
    return out[0] if values.ndim == 1 else out
