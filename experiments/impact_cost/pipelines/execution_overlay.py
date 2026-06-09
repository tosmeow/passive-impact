"""Plot passive execution times using raw first-level depth data."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.anchored_simulator import select_passive_limit_flags
from ..core.cost_utils import event_seconds, track_passive_fills
from ..core.level_execution import load_first_level_execution_window


@dataclass(frozen=True)
class ExecutionOverlayConfig:
    """Configuration for plotting passive postings/fills on a raw q1 path.

    This is a visual diagnostic for one start time. It selects passive orders
    every `every_seconds` until `selection_stop_seconds`, then compares a short
    horizon plot with a longer until-filled view.
    """

    raw_level_path: str = "experiments/impact_cost/data/raw/2025_05_29_ESM5.parquet"
    output_dir: str = "experiments/impact_cost/runs/execution_overlay/passive_1s_stop15"
    image_dir: str = "experiments/impact_cost/runs/execution_overlay/images"
    start_time: str = "2025-05-29T11:00:00-04:00"
    horizon_seconds: float = 30.0
    tracking_horizon_seconds: float = 900.0
    every_seconds: float = 1.0
    selection_stop_seconds: float = 15.0
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    cancellation_policy: str = "top"
    theta: float = 1.0
    seed: int = 2027


def run_execution_overlay(cfg: ExecutionOverlayConfig) -> dict[str, Any]:
    """Build passive execution overlay CSVs/plots for one raw-depth window."""
    output_dir = Path(cfg.output_dir)
    image_dir = Path(cfg.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    short = _build_execution_result(cfg, cfg.horizon_seconds)
    long = _build_execution_result(cfg, cfg.tracking_horizon_seconds)

    short_csv = output_dir / "passive_execution_times.csv"
    long_csv = output_dir / "passive_execution_times_until_filled.csv"
    _write_orders(short["orders"], short_csv)
    _write_orders(long["orders"], long_csv)

    short_plot = image_dir / "passive_execution_overlay_1s_stop15.png"
    long_plot = image_dir / "passive_execution_overlay_until_filled_1s_stop15.png"
    _plot_overlay(
        short_plot,
        short["window"],
        short["seconds"],
        short["orders"],
        q1_col=short["q1_col"],
        raw_side=cfg.raw_side,
        horizon_seconds=cfg.horizon_seconds,
        title="Passive postings/executions, start 11:00, 30s horizon with raw q1",
    )
    max_complete = long["orders"]["completed_time_s"].dropna().max()
    long_xmax = min(
        cfg.tracking_horizon_seconds,
        max(cfg.horizon_seconds, float(max_complete) + 5.0 if pd.notna(max_complete) else cfg.horizon_seconds),
    )
    _plot_overlay(
        long_plot,
        long["window"],
        long["seconds"],
        long["orders"],
        q1_col=long["q1_col"],
        raw_side=cfg.raw_side,
        horizon_seconds=long_xmax,
        title="Passive postings/executions, start 11:00, tracked until filled with raw q1",
    )
    _copy_plot(short_plot, output_dir / "passive_execution_overlay.png")
    _copy_plot(long_plot, output_dir / "passive_execution_overlay_until_filled.png")

    with open(output_dir / "execution_overlay_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg) | {"q1_col": long["q1_col"]}, f, indent=2)

    return {
        "short_plot": short_plot,
        "long_plot": long_plot,
        "short_csv": short_csv,
        "long_csv": long_csv,
        "q1_col": long["q1_col"],
        "n_orders": int(len(long["orders"])),
        "n_completed_in_30s": int(short["orders"]["completed_time_s"].notna().sum()),
        "n_completed_until_filled": int(long["orders"]["completed_time_s"].notna().sum()),
    }


def _build_execution_result(cfg: ExecutionOverlayConfig, horizon_seconds: float) -> dict[str, Any]:
    window, origin, q1_col = load_first_level_execution_window(
        cfg.raw_level_path,
        start_time=cfg.start_time,
        horizon_seconds=horizon_seconds,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
    )
    seconds = event_seconds(window, origin=origin)
    flags = select_passive_limit_flags(
        window,
        "first_every",
        raw_side=cfg.raw_side,
        level_col="level",
        target_level=1,
        every_seconds=cfg.every_seconds,
        seed=cfg.seed,
        origin=origin,
        ts_col="ts",
    )
    flags &= seconds <= cfg.selection_stop_seconds
    result = track_passive_fills(
        window,
        flags,
        side=cfg.raw_side,
        market_side=cfg.market_side,
        queue_col="q1",
        level_col="level",
        target_level=1,
        cancellation_policy=cfg.cancellation_policy,
        theta=cfg.theta,
        seed=cfg.seed,
    )
    orders = result.orders.copy()
    if orders.empty:
        orders["post_time_s"] = pd.Series(dtype=np.float64)
        orders["q1_post"] = pd.Series(dtype=np.float64)
        orders["completed_time_s"] = pd.Series(dtype=np.float64)
        orders["latency_s"] = pd.Series(dtype=np.float64)
        return {"window": window, "seconds": seconds, "orders": orders, "q1_col": q1_col}

    passive_rows = window.iloc[orders["row_pos"].to_numpy(dtype=np.int64)].reset_index(drop=True)
    orders["source_row_pos"] = passive_rows["source_row_pos"].to_numpy(dtype=np.int64)
    orders["post_time_s"] = event_seconds(passive_rows, origin=origin)
    orders["post_ts"] = passive_rows["ts"].astype(str).to_numpy()
    orders["post_qty"] = passive_rows["qty"].to_numpy(dtype=np.int64)
    orders["source_qty"] = passive_rows["source_qty"].to_numpy(dtype=np.int64)
    orders["q1_post"] = passive_rows["q1"].to_numpy(dtype=np.float64)
    orders["completed_time_s"] = orders["completed_time"].astype(float)
    orders["completed_ts"] = [
        str(origin + pd.Timedelta(seconds=float(t))) if pd.notna(t) else ""
        for t in orders["completed_time_s"]
    ]
    orders["latency_s"] = orders["completed_time_s"] - orders["post_time_s"]
    return {"window": window, "seconds": seconds, "orders": orders, "q1_col": q1_col}


def _write_orders(orders: pd.DataFrame, path: Path) -> None:
    cols = [
        "order_id",
        "row_pos",
        "source_row_pos",
        "l_index",
        "post_time_s",
        "post_ts",
        "post_qty",
        "source_qty",
        "q1_post",
        "executed_qty",
        "remaining_qty",
        "completed_time_s",
        "completed_ts",
        "latency_s",
    ]
    orders[[col for col in cols if col in orders.columns]].to_csv(path, index=False)


def _plot_overlay(
    path: Path,
    window: pd.DataFrame,
    seconds: np.ndarray,
    orders: pd.DataFrame,
    *,
    q1_col: str,
    raw_side: str,
    horizon_seconds: float,
    title: str,
) -> None:
    mpl_cache = Path("/private/tmp/matplotlib-cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q1 = window["q1"].to_numpy(dtype=np.float64)
    plot_mask = seconds <= horizon_seconds
    fig, (ax_q, ax_life) = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.35, 1.35]},
    )
    q1_label = f"raw q1 posting side {raw_side} ({q1_col})"
    ax_q.step(seconds[plot_mask], q1[plot_mask], where="post", color="black", linewidth=0.9, label=q1_label)
    if not orders.empty:
        ax_q.scatter(
            orders["post_time_s"],
            orders["q1_post"],
            color="#b83232",
            s=34,
            zorder=5,
            label="posted passive L",
        )
        completed = orders.dropna(subset=["completed_time_s"])
        if not completed.empty:
            exec_q = np.interp(completed["completed_time_s"], seconds, q1)
            ax_q.scatter(
                completed["completed_time_s"],
                exec_q,
                color="#1f7a3d",
                s=38,
                marker="D",
                zorder=6,
                label="final execution",
            )
        for idx, row in orders.iterrows():
            ax_q.axvline(row["post_time_s"], color="#b83232", alpha=0.14, linewidth=0.8)
            label_y = row["q1_post"] + (2.0 if idx % 2 == 0 else -3.0)
            ax_q.annotate(
                f"q1={int(row['q1_post'])}",
                xy=(row["post_time_s"], row["q1_post"]),
                xytext=(row["post_time_s"] + 0.18, label_y),
                fontsize=7.5,
                color="#7b1f1f",
                arrowprops={"arrowstyle": "-", "color": "#b83232", "alpha": 0.35, "lw": 0.6},
            )
            if pd.notna(row["completed_time_s"]):
                ax_q.axvline(row["completed_time_s"], color="#1f7a3d", alpha=0.16, linewidth=0.8)

        for _, row in orders.sort_values("order_id", ascending=False).iterrows():
            y = int(row["order_id"])
            if pd.notna(row["completed_time_s"]) and row["completed_time_s"] <= horizon_seconds:
                end = row["completed_time_s"]
                color = "#557aa6"
                status = f"{row['latency_s']:.1f}s"
                ax_life.scatter(end, y, color="#1f7a3d", marker="D", s=28, zorder=5)
            else:
                end = horizon_seconds
                color = "#a66a55"
                status = "open"
            ax_life.hlines(y, row["post_time_s"], end, color=color, linewidth=2.0, alpha=0.85)
            ax_life.scatter(row["post_time_s"], y, color="#b83232", s=26, zorder=4)
            ax_life.text(
                min(end + 0.35, horizon_seconds + 0.35),
                y,
                f"{status}  q1={int(row['q1_post'])}",
                va="center",
                fontsize=8,
                color="0.25",
            )

    ax_q.set_title(title)
    ax_q.set_ylabel(q1_label)
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="best")
    ax_life.set_ylabel("order id")
    ax_life.set_xlabel("seconds from start")
    if not orders.empty:
        ax_life.set_yticks(orders["order_id"].astype(int).to_numpy())
        ax_life.set_ylim(-0.8, len(orders) - 0.2)
    ax_life.set_xlim(0.0, horizon_seconds)
    ax_life.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _copy_plot(source: Path, destination: Path) -> None:
    destination.write_bytes(source.read_bytes())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-level-path", default=ExecutionOverlayConfig.raw_level_path)
    parser.add_argument("--output-dir", default=ExecutionOverlayConfig.output_dir)
    parser.add_argument("--image-dir", default=ExecutionOverlayConfig.image_dir)
    parser.add_argument("--start-time", default=ExecutionOverlayConfig.start_time)
    parser.add_argument("--horizon-seconds", type=float, default=ExecutionOverlayConfig.horizon_seconds)
    parser.add_argument("--tracking-horizon-seconds", type=float, default=ExecutionOverlayConfig.tracking_horizon_seconds)
    parser.add_argument("--every-seconds", type=float, default=ExecutionOverlayConfig.every_seconds)
    parser.add_argument("--selection-stop-seconds", type=float, default=ExecutionOverlayConfig.selection_stop_seconds)
    parser.add_argument("--raw-side", default=ExecutionOverlayConfig.raw_side)
    parser.add_argument("--queue-col", default=ExecutionOverlayConfig.queue_col)
    parser.add_argument("--market-side", default=ExecutionOverlayConfig.market_side)
    parser.add_argument("--cancellation-policy", default=ExecutionOverlayConfig.cancellation_policy)
    parser.add_argument("--theta", type=float, default=ExecutionOverlayConfig.theta)
    parser.add_argument("--seed", type=int, default=ExecutionOverlayConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ExecutionOverlayConfig(**vars(args))
    summary = run_execution_overlay(cfg)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
