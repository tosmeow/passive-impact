"""Resolve synthetic passive lifecycle intentions onto empirical event rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .cost_utils import CANCEL, LIMIT, event_seconds


BASE_ORDER_COLUMNS = [
    "episode_id",
    "policy_path_id",
    "cycle_id",
    "order_id",
    "order_slot",
    "post_time_s",
    "qty",
    "filled",
    "fill_time_s",
    "cancel_time_s",
    "terminal_time_s",
    "terminal_action",
]
ORDER_COLUMNS = BASE_ORDER_COLUMNS + [
    "requested_post_time_s",
    "last_post_time_s",
    "posted_qty",
    "unresolved_post_qty",
    "resolved_source_rows",
    "resolved",
]
FILL_COLUMNS = BASE_ORDER_COLUMNS + [
    "requested_post_time_s",
    "requested_fill_time_s",
]
CANCEL_COLUMNS = BASE_ORDER_COLUMNS + [
    "requested_post_time_s",
    "requested_cancel_time_s",
    "resolved_source_rows",
]
EVENT_COLUMNS = BASE_ORDER_COLUMNS + [
    "requested_time_s",
    "time_s",
    "event_kind",
    "displacement_delta",
    "source_row_pos",
    "row_pos",
]
UNRESOLVED_COLUMNS = BASE_ORDER_COLUMNS + [
    "event_kind",
    "requested_time_s",
    "requested_qty",
    "resolved_qty",
    "unresolved_qty",
]


@dataclass
class _Allocation:
    row_pos: int
    source_row_pos: int
    time_s: float
    qty: int


def resolve_lifecycle_to_observed_rows(
    window: pd.DataFrame,
    lifecycle: dict[str, pd.DataFrame],
    *,
    raw_side: str,
    origin: object,
    horizon_seconds: float,
    qty_col: str = "qty",
    ts_col: str = "ts",
    order_type_col: str = "order_type",
    side_col: str = "side",
) -> dict[str, object]:
    """Snap generated post/cancel intentions to observed limit/cancel rows.

    Fill/execution times remain the random lifecycle times. Only displayed
    quantity changes that should exist inside the factual queue are resolved
    onto empirical rows.
    """
    seconds = event_seconds(window, ts_col=ts_col, origin=origin)
    types = window[order_type_col].astype(str).str.lower().to_numpy()
    sides = window[side_col].astype(str).to_numpy()
    qty = window[qty_col].to_numpy(dtype=np.int64)
    if np.any(qty < 0):
        raise ValueError("qty must be non-negative")

    source_rows = (
        window["source_row_pos"].to_numpy(dtype=np.int64)
        if "source_row_pos" in window.columns
        else np.arange(len(window), dtype=np.int64)
    )
    posting_side = str(raw_side).upper()
    available = qty.copy()
    own_qtys = np.zeros(len(window), dtype=np.int64)

    desired_orders = lifecycle["orders"].copy()
    desired_fills = lifecycle["fills"].copy()
    desired_cancels = lifecycle["cancels"].copy()
    desired_cycles = lifecycle["cycle_summary"].copy()

    order_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    cancel_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    fill_by_order = (
        desired_fills.set_index("order_id").to_dict("index")
        if not desired_fills.empty
        else {}
    )
    cancel_by_order = (
        desired_cancels.set_index("order_id").to_dict("index")
        if not desired_cancels.empty
        else {}
    )

    for desired in desired_orders.sort_values("post_time_s").itertuples(index=False):
        base = _base_order_dict(desired)
        requested_qty = int(base["qty"])
        requested_post_time = float(base["post_time_s"])
        post_allocs = _allocate_units(
            seconds=seconds,
            types=types,
            sides=sides,
            source_rows=source_rows,
            available=available,
            event_type=LIMIT,
            side=posting_side,
            after_time_s=requested_post_time,
            requested_qty=requested_qty,
            horizon_seconds=float(horizon_seconds),
        )
        posted_qty = int(sum(alloc.qty for alloc in post_allocs))
        for alloc in post_allocs:
            own_qtys[alloc.row_pos] += alloc.qty
            event_rows.append(
                base
                | {
                    "requested_time_s": requested_post_time,
                    "time_s": alloc.time_s,
                    "qty": alloc.qty,
                    "event_kind": "post",
                    "displacement_delta": alloc.qty,
                    "source_row_pos": alloc.source_row_pos,
                    "row_pos": alloc.row_pos,
                }
            )

        if posted_qty < requested_qty:
            unresolved_rows.append(
                base
                | {
                    "event_kind": "post",
                    "requested_time_s": requested_post_time,
                    "requested_qty": requested_qty,
                    "resolved_qty": posted_qty,
                    "unresolved_qty": requested_qty - posted_qty,
                }
            )

        if posted_qty <= 0:
            continue

        first_post = min(alloc.time_s for alloc in post_allocs)
        last_post = max(alloc.time_s for alloc in post_allocs)
        resolved_sources = ",".join(str(alloc.source_row_pos) for alloc in post_allocs)
        order_rows.append(
            base
            | {
                "requested_post_time_s": requested_post_time,
                "post_time_s": first_post,
                "last_post_time_s": last_post,
                "qty": posted_qty,
                "posted_qty": posted_qty,
                "unresolved_post_qty": requested_qty - posted_qty,
                "resolved_source_rows": resolved_sources,
                "resolved": posted_qty == requested_qty,
            }
        )

        order_id = int(base["order_id"])
        fill_info = fill_by_order.get(order_id)
        if fill_info is not None:
            requested_fill_time = float(fill_info["fill_time_s"])
            fill_time = max(requested_fill_time, last_post)
            if fill_time <= float(horizon_seconds):
                fill_qty = posted_qty
                fill_rows.append(
                    base
                    | {
                        "post_time_s": first_post,
                        "requested_post_time_s": requested_post_time,
                        "requested_fill_time_s": requested_fill_time,
                        "fill_time_s": fill_time,
                        "qty": fill_qty,
                    }
                )
                event_rows.append(
                    base
                    | {
                        "post_time_s": first_post,
                        "requested_time_s": requested_fill_time,
                        "time_s": fill_time,
                        "qty": fill_qty,
                        "event_kind": "fill",
                        "displacement_delta": -fill_qty,
                        "source_row_pos": -1,
                        "row_pos": -1,
                    }
                )
            continue

        cancel_info = cancel_by_order.get(order_id)
        if cancel_info is None:
            continue

        requested_cancel_time = float(cancel_info["cancel_time_s"])
        cancel_allocs = _allocate_units(
            seconds=seconds,
            types=types,
            sides=sides,
            source_rows=source_rows,
            available=available,
            event_type=CANCEL,
            side=posting_side,
            after_time_s=max(requested_cancel_time, last_post),
            requested_qty=posted_qty,
            horizon_seconds=float(horizon_seconds),
        )
        canceled_qty = int(sum(alloc.qty for alloc in cancel_allocs))
        for alloc in cancel_allocs:
            own_qtys[alloc.row_pos] += alloc.qty
            event_rows.append(
                base
                | {
                    "post_time_s": first_post,
                    "requested_time_s": requested_cancel_time,
                    "time_s": alloc.time_s,
                    "qty": alloc.qty,
                    "event_kind": "cancel",
                    "displacement_delta": -alloc.qty,
                    "source_row_pos": alloc.source_row_pos,
                    "row_pos": alloc.row_pos,
                }
            )
        if canceled_qty > 0:
            cancel_rows.append(
                base
                | {
                    "post_time_s": first_post,
                    "requested_post_time_s": requested_post_time,
                    "requested_cancel_time_s": requested_cancel_time,
                    "cancel_time_s": max(alloc.time_s for alloc in cancel_allocs),
                    "qty": canceled_qty,
                    "resolved_source_rows": ",".join(
                        str(alloc.source_row_pos) for alloc in cancel_allocs
                    ),
                }
            )
        if canceled_qty < posted_qty:
            unresolved_rows.append(
                base
                | {
                    "event_kind": "cancel",
                    "requested_time_s": requested_cancel_time,
                    "requested_qty": posted_qty,
                    "resolved_qty": canceled_qty,
                    "unresolved_qty": posted_qty - canceled_qty,
                }
            )

    orders = _frame(order_rows, ORDER_COLUMNS)
    fills = _frame(fill_rows, FILL_COLUMNS)
    cancels = _frame(cancel_rows, CANCEL_COLUMNS)
    events = _frame(event_rows, EVENT_COLUMNS)
    unresolved = _frame(unresolved_rows, UNRESOLVED_COLUMNS)
    return {
        "orders": orders,
        "fills": fills,
        "cancels": cancels,
        "events": events,
        "cycle_summary": desired_cycles,
        "own_qtys": own_qtys,
        "unresolved": unresolved,
    }


def _allocate_units(
    *,
    seconds: np.ndarray,
    types: np.ndarray,
    sides: np.ndarray,
    source_rows: np.ndarray,
    available: np.ndarray,
    event_type: str,
    side: str,
    after_time_s: float,
    requested_qty: int,
    horizon_seconds: float,
) -> list[_Allocation]:
    if requested_qty <= 0:
        return []
    remaining = int(requested_qty)
    out: list[_Allocation] = []
    eligible = np.flatnonzero(
        (types == event_type)
        & (sides == side)
        & (seconds >= float(after_time_s))
        & (seconds <= float(horizon_seconds))
        & (available > 0)
    )
    for row_pos in eligible:
        take = int(min(available[row_pos], remaining))
        if take <= 0:
            continue
        available[row_pos] -= take
        remaining -= take
        out.append(
            _Allocation(
                row_pos=int(row_pos),
                source_row_pos=int(source_rows[row_pos]),
                time_s=float(seconds[row_pos]),
                qty=take,
            )
        )
        if remaining == 0:
            break
    return out


def _base_order_dict(row: object) -> dict[str, object]:
    names = getattr(row, "_fields")
    return {name: getattr(row, name) for name in names}


def _frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    extras = [col for col in out.columns if col not in columns]
    return out[columns + extras]
