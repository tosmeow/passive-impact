"""Queue reconstruction helpers for impact-cost data checks."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .level_execution import market_side_for_queue


def event_delta(order_type: str, qty: int) -> int:
    """Return the simple queue delta implied by one event row."""
    typ = str(order_type).lower()
    if typ == "limit":
        return int(qty)
    if typ in {"cancel", "market"}:
        return -int(qty)
    return 0


def replay_consistency_report(
    window: pd.DataFrame,
    *,
    raw_side: str,
    queue_col: str,
    initial_q: int,
    market_side: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare raw queue deltas to a simple limit/cancel/market replay."""
    consuming_side = market_side_for_queue(
        raw_side=raw_side,
        queue_col=queue_col,
        market_side=market_side,
    )
    report = window[["ts", "order_type", "side", "qty", queue_col]].copy()
    raw_delta = report[queue_col].diff()
    raw_delta.iloc[0] = float(report[queue_col].iloc[0] - initial_q)

    expected = []
    reconstructed = []
    reconstructed_q = int(initial_q)
    for row in report.itertuples(index=False):
        row_side = str(row.side)
        row_type = str(row.order_type).lower()
        prev_q = reconstructed_q
        if row_type in {"limit", "cancel"} and row_side == raw_side:
            if row_type == "limit":
                reconstructed_q += int(row.qty)
            else:
                reconstructed_q = max(0, reconstructed_q - int(row.qty))
        elif row_type == "market" and row_side == consuming_side:
            reconstructed_q = max(0, reconstructed_q - int(row.qty))
        expected.append(reconstructed_q - prev_q)
        reconstructed.append(reconstructed_q)
    expected = np.asarray(expected, dtype=np.float64)

    report["raw_delta"] = raw_delta.to_numpy(dtype=np.float64)
    report["expected_delta"] = expected
    report["residual"] = report["raw_delta"] - report["expected_delta"]
    report["reconstructed_queue"] = np.asarray(reconstructed, dtype=np.float64)
    report["level_diff"] = report[queue_col].to_numpy(dtype=np.float64) - report[
        "reconstructed_queue"
    ]

    abs_residual = report["residual"].abs()
    abs_level_diff = report["level_diff"].abs()
    summary = {
        "raw_net_delta": float(report["raw_delta"].sum()),
        "expected_net_delta": float(report["expected_delta"].sum()),
        "residual_net_delta": float(report["residual"].sum()),
        "nonzero_residual_rows": int((abs_residual > 0).sum()),
        "max_abs_residual": float(abs_residual.max()),
        "mean_abs_residual": float(abs_residual.mean()),
        "final_level_diff": float(report["level_diff"].iloc[-1]),
        "max_abs_level_diff": float(abs_level_diff.max()),
        "mean_abs_level_diff": float(abs_level_diff.mean()),
    }
    return report, summary


def infer_initial_queue(
    window: pd.DataFrame,
    *,
    raw_side: str,
    queue_col: str,
    market_side: str | None = None,
) -> int:
    """Infer the pre-window queue from the first post-event queue snapshot."""
    consuming_side = market_side_for_queue(
        raw_side=raw_side,
        queue_col=queue_col,
        market_side=market_side,
    )
    first = window.iloc[0]
    post_q = int(first[queue_col])
    typ = str(first["order_type"]).lower()
    event_side = consuming_side if typ == "market" else raw_side
    if str(first["side"]) != event_side:
        return post_q
    return max(0, post_q - event_delta(first["order_type"], int(first["qty"])))
