"""Per-order running cost and queue-priority diagnostics."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.level_execution import market_side_for_queue, q1_column_for_side


LIMIT_DIM = 0
CANCEL_DIM = 1
MARKET_DIM = 2
IGNORED_DIM = -1


@dataclass(frozen=True)
class RunningCostDiagnosticsConfig:
    """Configuration for per-order cost and priority-path diagnostic plots.

    `cost_output_dir` must contain the CSV files emitted by
    `impact_cost_pipeline`. The diagnostic reloads raw first-level depth to
    reconstruct queue/priority paths for selected orders.
    """

    raw_level_path: str = "experiments/impact_cost/data/raw/2025_05_29_ESM5.parquet"
    cost_output_dir: str = "experiments/impact_cost/runs/impact_cost_latency_le_30s"
    output_dir: str = "experiments/impact_cost/runs/running_cost_diagnostics"
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    cancellation_policy: str = "top"
    cap_position_by_queue_post: bool = True
    max_plots: int = 8
    order_by: str = "window_id"


def run_running_cost_diagnostics(cfg: RunningCostDiagnosticsConfig) -> dict[str, Any]:
    """Render per-order running-cost diagnostic plots and a manifest CSV."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orders, fills, contributions = _load_cost_outputs(Path(cfg.cost_output_dir))
    raw, arrays = _load_raw_arrays(cfg)
    selected = _select_orders_for_plots(orders, contributions, cfg)

    manifest_rows = []
    for row in selected.itertuples(index=False):
        order = orders[orders["order_id"].astype(int) == int(row.order_id)].iloc[0]
        order_fills = fills[fills["order_id"].astype(int) == int(row.order_id)].copy()
        order_contribs = contributions[
            contributions["order_id"].astype(int) == int(row.order_id)
        ].copy()
        if order_fills.empty or order_contribs.empty:
            continue

        priority = _priority_path(
            arrays,
            source_row_pos=int(order.source_row_pos),
            completed_time_s=float(order.completed_time_s),
            window_start=pd.Timestamp(order.window_start),
            policy=cfg.cancellation_policy,
            cap_position_by_queue_post=cfg.cap_position_by_queue_post,
        )
        q1_path = _q1_path(
            raw,
            arrays,
            window_start=pd.Timestamp(order.window_start),
            post_time_s=float(order.post_time_s),
            completed_time_s=float(order.completed_time_s),
        )
        running_cost = _running_cost_by_simulation(
            order_contribs,
            post_time_s=float(order.post_time_s),
            completed_time_s=float(order.completed_time_s),
        )

        path = output_dir / f"window_{int(order.window_id):03d}_order_{int(order.order_id):03d}.png"
        _plot_order_diagnostic(
            path,
            order=order,
            fills=order_fills,
            q1_path=q1_path,
            priority=priority,
            running_cost=running_cost,
            q1_col=arrays["q1_col"],
        )
        manifest_rows.append(
            {
                "window_id": int(order.window_id),
                "order_id": int(order.order_id),
                "window_start": str(order.window_start),
                "post_time_s": float(order.post_time_s),
                "completed_time_s": float(order.completed_time_s),
                "latency_s": float(order.latency_s),
                "post_qty": int(order.post_qty),
                "mean_final_cost": float(
                    running_cost.groupby("simulation")["running_cost"].last().mean()
                ),
                "plot_path": str(path),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    with open(output_dir / "running_cost_diagnostics_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return {
        "output_dir": output_dir,
        "manifest_path": output_dir / "manifest.csv",
        "n_plots": int(len(manifest)),
    }


def _load_cost_outputs(cost_output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(cost_output_dir / "selected_latency_orders.csv")
    fills = pd.read_csv(cost_output_dir / "selected_latency_fills.csv")
    contributions = pd.read_csv(cost_output_dir / "impact_cost_fill_contributions.csv")
    return orders, fills, contributions


def _select_orders_for_plots(
    orders: pd.DataFrame,
    contributions: pd.DataFrame,
    cfg: RunningCostDiagnosticsConfig,
) -> pd.DataFrame:
    final_cost = (
        contributions.groupby(["window_id", "order_id", "simulation"], sort=True)["contribution"]
        .sum()
        .groupby(["window_id", "order_id"])
        .mean()
    )
    selected = orders.copy()
    selected["mean_contribution"] = [
        float(final_cost.get((int(row.window_id), int(row.order_id)), 0.0))
        for row in selected.itertuples(index=False)
    ]
    order_by = cfg.order_by.lower()
    if order_by == "abs_cost":
        selected = selected.reindex(selected["mean_contribution"].abs().sort_values(ascending=False).index)
    elif order_by == "latency":
        selected = selected.sort_values("latency_s", ascending=False)
    elif order_by == "window_id":
        selected = selected.sort_values(["window_id", "order_id"])
    else:
        raise ValueError("order_by must be 'window_id', 'latency', or 'abs_cost'")
    return selected.head(int(cfg.max_plots)).copy()


def _load_raw_arrays(cfg: RunningCostDiagnosticsConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    q1_col = q1_column_for_side(raw_side=cfg.raw_side, queue_col=cfg.queue_col)
    consuming_side = market_side_for_queue(
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
    )
    columns = ["ts", "order_type", "side", "qty", q1_col]
    raw = pd.read_parquet(cfg.raw_level_path, columns=columns)
    raw = raw.sort_values("ts", kind="mergesort").reset_index(drop=True)
    raw["source_row_pos"] = np.arange(len(raw), dtype=np.int64)

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
    for typ, dim, sign in (
        ("limit", LIMIT_DIM, 1.0),
        ("cancel", CANCEL_DIM, -1.0),
        ("market", MARKET_DIM, -1.0),
    ):
        event_side = consuming_side if typ == "market" else cfg.raw_side
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

    return raw, {
        "origin": origin,
        "times": times,
        "q1_post": q1_post,
        "dims": dims,
        "qty": qty,
        "q1_col": q1_col,
        "market_side": consuming_side,
    }


def _q1_path(
    raw: pd.DataFrame,
    arrays: dict[str, Any],
    *,
    window_start: pd.Timestamp,
    post_time_s: float,
    completed_time_s: float,
) -> pd.DataFrame:
    origin = arrays["origin"]
    start_s = (window_start - origin).total_seconds() + post_time_s
    end_s = (window_start - origin).total_seconds() + completed_time_s
    lo = int(np.searchsorted(arrays["times"], start_s, side="left"))
    hi = int(np.searchsorted(arrays["times"], end_s, side="right"))
    return pd.DataFrame(
        {
            "time_s": arrays["times"][lo:hi] - (window_start - origin).total_seconds(),
            "q1": arrays["q1_post"][lo:hi],
            "source_row_pos": raw["source_row_pos"].to_numpy(dtype=np.int64)[lo:hi],
        }
    )


def _priority_path(
    arrays: dict[str, Any],
    *,
    source_row_pos: int,
    completed_time_s: float,
    window_start: pd.Timestamp,
    policy: str,
    cap_position_by_queue_post: bool,
) -> pd.DataFrame:
    origin = arrays["origin"]
    window_start_s = (window_start - origin).total_seconds()
    start_idx = source_row_pos
    end_time_abs = window_start_s + completed_time_s
    end_idx = int(np.searchsorted(arrays["times"], end_time_abs, side="right"))

    remaining = 0
    position = 0.0
    top_qty = 0
    records = []
    for idx in range(start_idx, end_idx):
        dim = int(arrays["dims"][idx])
        event_qty = int(arrays["qty"][idx])
        if dim < 0 or event_qty == 0:
            continue
        time_s = float(arrays["times"][idx] - window_start_s)
        is_own = idx == source_row_pos
        fill_qty = 0

        if dim == LIMIT_DIM:
            if is_own:
                remaining = event_qty
                position = max(float(arrays["q1_post"][idx]), float(remaining))
            elif remaining > 0:
                top_qty += event_qty
        elif dim == MARKET_DIM and remaining > 0:
            fill_qty = _market_fill_qty(position, remaining, event_qty)
            remaining -= fill_qty
            position = 0.0 if remaining == 0 else max(position - event_qty, float(remaining))
        elif dim == CANCEL_DIM and remaining > 0:
            desired_top = event_qty if policy == "top" else 0
            cancel_top = min(top_qty, desired_top)
            top_qty -= cancel_top
            cancel_position = event_qty - cancel_top
            position = max(position - cancel_position, float(remaining))

        if cap_position_by_queue_post and remaining > 0:
            position = max(min(position, float(arrays["q1_post"][idx])), float(remaining))
            top_qty = min(top_qty, int(max(position - remaining, 0.0)))

        if is_own or remaining > 0 or fill_qty > 0:
            records.append(
                {
                    "time_s": time_s,
                    "position_qty": position,
                    "remaining_qty": remaining,
                    "top_qty": top_qty,
                    "event_dim": dim,
                    "event_qty": event_qty,
                    "fill_qty": fill_qty,
                    "q1": float(arrays["q1_post"][idx]),
                }
            )
        if remaining == 0 and idx > source_row_pos:
            break

    return pd.DataFrame(records)


def _market_fill_qty(position: float, remaining: int, event_qty: int) -> int:
    ahead_before = max(position - float(remaining), 0.0)
    consumed_until = min(position, float(event_qty))
    return int(max(consumed_until - ahead_before, 0.0))


def _running_cost_by_simulation(
    contributions: pd.DataFrame,
    *,
    post_time_s: float,
    completed_time_s: float,
) -> pd.DataFrame:
    rows = []
    simulations = sorted(int(x) for x in contributions["simulation"].unique())
    for sim in simulations:
        sim_rows = contributions[contributions["simulation"].astype(int) == sim].sort_values(
            "fill_time_s"
        )
        running = 0.0
        rows.append({"simulation": sim, "time_s": post_time_s, "running_cost": 0.0})
        for fill in sim_rows.itertuples(index=False):
            running += float(fill.contribution)
            rows.append(
                {
                    "simulation": sim,
                    "time_s": float(fill.fill_time_s),
                    "running_cost": running,
                }
            )
        if not sim_rows.empty and float(sim_rows["fill_time_s"].iloc[-1]) < completed_time_s:
            rows.append(
                {
                    "simulation": sim,
                    "time_s": completed_time_s,
                    "running_cost": running,
                }
            )
    return pd.DataFrame(rows)


def _plot_order_diagnostic(
    path: Path,
    *,
    order: Any,
    fills: pd.DataFrame,
    q1_path: pd.DataFrame,
    priority: pd.DataFrame,
    running_cost: pd.DataFrame,
    q1_col: str,
) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, (ax_cost, ax_queue) = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=False)

    for _, sim in running_cost.groupby("simulation", sort=True):
        ax_cost.step(
            sim["time_s"],
            sim["running_cost"],
            where="post",
            color="0.72",
            alpha=0.45,
            linewidth=0.8,
        )
    summary = _running_cost_summary(running_cost)
    if not summary.empty:
        ax_cost.fill_between(
            summary["time_s"],
            summary["q05"],
            summary["q95"],
            step="post",
            color="#5b8cc0",
            alpha=0.18,
            linewidth=0,
        )
        ax_cost.step(
            summary["time_s"],
            summary["mean"],
            where="post",
            color="#1f5d99",
            linewidth=2.0,
            label="mean running cost",
        )
    for fill in fills.itertuples(index=False):
        ax_cost.axvline(float(fill.fill_time_s), color="#b83232", linewidth=0.9, alpha=0.65)
    ax_cost.axhline(0.0, color="0.25", linewidth=0.8)
    ax_cost.set_title("Running execution cost")
    ax_cost.set_xlabel("seconds from minute start")
    ax_cost.set_ylabel("price units x contracts")
    ax_cost.grid(True, alpha=0.25)
    ax_cost.legend(loc="best")

    ax_queue.plot(q1_path["time_s"], q1_path["q1"], color="black", linewidth=1.1, label=q1_col)
    if not priority.empty:
        ax_queue.step(
            priority["time_s"],
            priority["position_qty"],
            where="post",
            color="#b83232",
            linewidth=1.6,
            label="own priority position",
        )
        fill_rows = priority[priority["fill_qty"] > 0]
        if not fill_rows.empty:
            ax_queue.scatter(
                fill_rows["time_s"],
                fill_rows["position_qty"],
                color="#b83232",
                edgecolor="white",
                linewidth=0.5,
                s=42,
                zorder=5,
                label="fill event",
            )
    ax_queue.axvline(float(order.post_time_s), color="#4b6f44", linewidth=1.0, linestyle="--")
    ax_queue.axvline(float(order.completed_time_s), color="#b83232", linewidth=1.0, linestyle="--")
    ax_queue.set_title("First queue and priority")
    ax_queue.set_xlabel("seconds from minute start")
    ax_queue.set_ylabel("queue units")
    ax_queue.grid(True, alpha=0.25)
    ax_queue.legend(loc="best")

    for ax in (ax_cost, ax_queue):
        ax.set_xlim(float(order.post_time_s), float(order.completed_time_s))

    fig.suptitle(
        f"window {int(order.window_id)}, order {int(order.order_id)}, "
        f"latency {float(order.latency_s):.2f}s",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _running_cost_summary(running_cost: pd.DataFrame) -> pd.DataFrame:
    times = np.array(sorted(running_cost["time_s"].unique()), dtype=np.float64)
    rows = []
    for time in times:
        values = []
        for _, sim in running_cost.groupby("simulation", sort=False):
            past = sim[sim["time_s"] <= time]
            values.append(float(past["running_cost"].iloc[-1]) if not past.empty else 0.0)
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "time_s": time,
                "mean": float(arr.mean()),
                "q05": float(np.quantile(arr, 0.05)),
                "q95": float(np.quantile(arr, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _setup_matplotlib() -> None:
    mpl_cache = Path("/private/tmp/matplotlib-cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-level-path", default=RunningCostDiagnosticsConfig.raw_level_path)
    parser.add_argument("--cost-output-dir", default=RunningCostDiagnosticsConfig.cost_output_dir)
    parser.add_argument("--output-dir", default=RunningCostDiagnosticsConfig.output_dir)
    parser.add_argument("--raw-side", default=RunningCostDiagnosticsConfig.raw_side)
    parser.add_argument("--queue-col", default=RunningCostDiagnosticsConfig.queue_col)
    parser.add_argument("--market-side", default=RunningCostDiagnosticsConfig.market_side)
    parser.add_argument("--cancellation-policy", default=RunningCostDiagnosticsConfig.cancellation_policy)
    parser.add_argument(
        "--no-cap-position-by-queue-post",
        dest="cap_position_by_queue_post",
        action="store_false",
    )
    parser.add_argument("--max-plots", type=int, default=RunningCostDiagnosticsConfig.max_plots)
    parser.add_argument(
        "--order-by",
        choices=["window_id", "latency", "abs_cost"],
        default=RunningCostDiagnosticsConfig.order_by,
    )
    return parser.parse_args()


def main() -> None:
    cfg = RunningCostDiagnosticsConfig(**vars(_parse_args()))
    summary = run_running_cost_diagnostics(cfg)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
