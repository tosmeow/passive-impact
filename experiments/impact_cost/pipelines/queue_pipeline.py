"""Parquet-backed conditional queue simulation pipeline for impact cost."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.anchored_simulator import (
    select_passive_limit_flags,
    simulate_anchored_queue_paths,
)
from ..core.cost_utils import (
    event_seconds,
)
from ..core.level_execution import market_side_for_queue


@dataclass(frozen=True)
class QueuePipelineConfig:
    """Configuration for anchored conditional queue-path simulations.

    Input rows must have `ts`, `order_type`, `side`, `qty`, and the selected
    queue column (`q_a` or `q_b`). The pipeline flags passive limit rows,
    simulates counterfactual no-us queue paths, and writes queue-path plots,
    CSV diagnostics, and JSON config into `output_dir`.
    """

    parquet_path: str = "experiments/impact_cost/data/processed/factual_2025_05_29_esm5.parquet"
    output_dir: str = "experiments/impact_cost/runs/queue_pipeline"
    horizon_seconds: float = 30.0
    every_seconds: float = 5.0
    n_simulations: int = 25
    n_grid: int = 600
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    selection_policy: str = "first_every"
    selection_fraction: float | None = None
    selection_indices: tuple[int, ...] = ()
    selection_index_base: int = 1
    selection_stop_seconds: float | None = None
    seed: int = 2027
    start_time: str | None = None
    a_l: float = 184.1372
    b_l: float = -0.0097
    a_c: float = 184.0456
    b_c: float = 0.00989
    require_replay_match: bool = False
    replay_tolerance: float = 0.0


def _load_window(cfg: QueuePipelineConfig) -> tuple[pd.DataFrame, pd.Timestamp]:
    columns = ["ts", "order_type", "side", "qty", "q_b", "q_a"]
    input_path = Path(cfg.parquet_path)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, usecols=columns)
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df = pd.read_parquet(input_path, columns=columns)
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    if df.empty:
        raise ValueError("input parquet has no rows")

    start = df["ts"].iloc[0] if cfg.start_time is None else pd.Timestamp(cfg.start_time)
    if getattr(df["ts"].dt, "tz", None) is not None and start.tzinfo is None:
        start = start.tz_localize(df["ts"].dt.tz)
    end = start + pd.Timedelta(seconds=cfg.horizon_seconds)
    window = df[(df["ts"] >= start) & (df["ts"] <= end)].copy().reset_index(drop=True)
    if window.empty:
        raise ValueError(f"no rows found in [{start}, {end}]")
    return window, start


def _event_delta(order_type: str, qty: int) -> int:
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
    """Compare raw queue deltas to the simple L/C/N replay model."""
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
    """Infer pre-window queue from the first post-event snapshot."""
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
    return max(0, post_q - _event_delta(first["order_type"], int(first["qty"])))


def _plot_queue_paths(
    path: Path,
    grid: np.ndarray,
    factual_queue: np.ndarray,
    conditioning_queue: np.ndarray,
    mechanical_no_us_queue: np.ndarray,
    simulated_queues: np.ndarray,
    own_order_times: np.ndarray,
    cfg: QueuePipelineConfig,
) -> None:
    mpl_cache = Path("/private/tmp/matplotlib-cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx in range(simulated_queues.shape[1]):
        ax.plot(grid, simulated_queues[:, idx], color="0.72", linewidth=0.8, alpha=0.45)

    mean_path = simulated_queues.mean(axis=1)
    q05, q95 = np.quantile(simulated_queues, [0.05, 0.95], axis=1)
    ax.fill_between(grid, q05, q95, color="#5b8cc0", alpha=0.18, linewidth=0)
    ax.plot(grid, mean_path, color="#1f5d99", linewidth=2.0, label="conditional mean")
    ax.plot(grid, factual_queue, color="black", linewidth=1.4, label="raw factual queue")
    ax.plot(
        grid,
        conditioning_queue,
        color="#8a5a00",
        linewidth=1.4,
        linestyle="--",
        label="anchored q_bar",
    )
    ax.plot(
        grid,
        mechanical_no_us_queue,
        color="#627a3d",
        linewidth=1.1,
        linestyle=":",
        label="mechanical no-us",
    )

    if len(own_order_times) > 0:
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            own_order_times,
            ymin=ymin,
            ymax=ymin + 0.08 * (ymax - ymin),
            color="#b83232",
            linewidth=1.0,
            label="flagged passive L",
        )
        ax.set_ylim(ymin, ymax)

    ax.set_title(
        f"Conditional queue simulations, post={cfg.raw_side}, market={cfg.market_side}, "
        f"horizon={cfg.horizon_seconds:g}s"
    )
    ax.set_xlabel("seconds from window start")
    ax.set_ylabel(cfg.queue_col)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_dq_paths(
    path: Path,
    grid: np.ndarray,
    simulated_offsets: np.ndarray,
    mechanical_offset: np.ndarray,
    own_order_times: np.ndarray,
    cfg: QueuePipelineConfig,
) -> None:
    mpl_cache = Path("/private/tmp/matplotlib-cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.4))
    for idx in range(simulated_offsets.shape[1]):
        ax.plot(grid, simulated_offsets[:, idx], color="0.72", linewidth=0.8, alpha=0.45)

    mean_path = simulated_offsets.mean(axis=1)
    q05, q95 = np.quantile(simulated_offsets, [0.05, 0.95], axis=1)
    ax.fill_between(grid, q05, q95, color="#5b8cc0", alpha=0.18, linewidth=0)
    ax.plot(grid, mean_path, color="#1f5d99", linewidth=2.0, label="conditional mean d_q")
    ax.plot(
        grid,
        mechanical_offset,
        color="#627a3d",
        linewidth=1.3,
        linestyle=":",
        label="mechanical no-us d_q",
    )
    ax.axhline(0.0, color="0.25", linewidth=0.8)

    if len(own_order_times) > 0:
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            own_order_times,
            ymin=ymin,
            ymax=ymin + 0.08 * (ymax - ymin),
            color="#b83232",
            linewidth=1.0,
            label="flagged passive L",
        )
        ax.set_ylim(ymin, ymax)

    ax.set_title(f"Conditional d_q = q - q_bar, post={cfg.raw_side}, market={cfg.market_side}")
    ax.set_xlabel("seconds from window start")
    ax.set_ylabel("d_q")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_replay_divergence(
    path: Path,
    replay_report: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    queue_col: str,
) -> None:
    mpl_cache = Path("/private/tmp/matplotlib-cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seconds = event_seconds(replay_report, origin=origin)
    fig, (ax_path, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax_path.plot(seconds, replay_report[queue_col], color="black", linewidth=1.2, label="raw factual queue")
    ax_path.plot(
        seconds,
        replay_report["reconstructed_queue"],
        color="#1f5d99",
        linewidth=1.2,
        label="reconstructed L-C-N queue",
    )
    ax_path.set_ylabel(queue_col)
    ax_path.legend(loc="best")
    ax_path.grid(True, alpha=0.25)

    ax_diff.axhline(0.0, color="0.25", linewidth=0.8)
    ax_diff.plot(seconds, replay_report["level_diff"], color="#b83232", linewidth=1.1)
    ax_diff.set_xlabel("seconds from window start")
    ax_diff.set_ylabel("raw - reconstructed")
    ax_diff.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_queue_pipeline(cfg: QueuePipelineConfig) -> dict[str, Any]:
    """Run the anchored queue-path experiment and return output paths/summary."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window, origin = _load_window(cfg)
    consuming_side = market_side_for_queue(
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
    )
    initial_q = infer_initial_queue(
        window,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=consuming_side,
    )
    replay_report, replay_summary = replay_consistency_report(
        window,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        initial_q=initial_q,
        market_side=consuming_side,
    )
    if (
        cfg.require_replay_match
        and replay_summary["max_abs_residual"] > cfg.replay_tolerance
    ):
        raise ValueError(
            "raw queue snapshots are not reproduced by simple L/C/N replay: "
            f"max_abs_residual={replay_summary['max_abs_residual']} "
            f"> tolerance={cfg.replay_tolerance}. "
            "Inspect replay_consistency.csv or choose a more stable window."
        )

    passive_flags = select_passive_limit_flags(
        window,
        cfg.selection_policy,
        raw_side=cfg.raw_side,
        market_side=consuming_side,
        every_seconds=cfg.every_seconds,
        fraction=cfg.selection_fraction,
        indices=cfg.selection_indices,
        index_base=cfg.selection_index_base,
        seed=cfg.seed,
        origin=origin,
        ts_col="ts",
    )
    if cfg.selection_stop_seconds is not None:
        if cfg.selection_stop_seconds < 0:
            raise ValueError("selection_stop_seconds must be non-negative")
        selection_seconds = event_seconds(window, origin=origin)
        passive_flags &= selection_seconds <= cfg.selection_stop_seconds
    passive_df = window[passive_flags].copy()

    grid = np.linspace(0.0, cfg.horizon_seconds, cfg.n_grid, dtype=np.float64)
    anchored = simulate_anchored_queue_paths(
        window,
        passive_flags,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=consuming_side,
        initial_q=initial_q,
        horizon_seconds=cfg.horizon_seconds,
        grid=grid,
        n_simulations=cfg.n_simulations,
        seed=cfg.seed,
        a_l=cfg.a_l,
        b_l=cfg.b_l,
        a_c=cfg.a_c,
        b_c=cfg.b_c,
        origin=origin,
    )
    factual_queue = anchored.factual_queue
    conditioning_queue = anchored.factual_queue
    mechanical_no_us_queue = anchored.mechanical_no_us_queue
    simulated_queues = anchored.simulated_queues
    simulated_offsets = anchored.simulated_offsets
    mechanical_offset = mechanical_no_us_queue - factual_queue

    own_order_row_times = (
        event_seconds(passive_df, origin=origin)
        if len(passive_df) > 0
        else np.array([], dtype=np.float64)
    )
    own_order_times = own_order_row_times
    own_orders = pd.DataFrame(
        {
            "row_pos": np.flatnonzero(passive_flags),
            "time": own_order_row_times,
            "qty": passive_df["qty"].to_numpy(dtype=np.int64),
            "ts": passive_df["ts"].astype(str).to_numpy(),
        }
    )

    np.save(output_dir / "times.npy", grid)
    np.save(output_dir / "factual_queue.npy", factual_queue)
    np.save(output_dir / "conditioning_queue.npy", conditioning_queue)
    np.save(output_dir / "mechanical_no_us_queue.npy", mechanical_no_us_queue)
    np.save(output_dir / "model_factual_queue.npy", conditioning_queue)
    np.save(output_dir / "simulated_queues.npy", simulated_queues)
    np.save(output_dir / "simulated_offsets.npy", simulated_offsets)
    np.save(output_dir / "mechanical_offset.npy", mechanical_offset)
    own_orders.to_csv(output_dir / "flagged_passive_limits.csv", index=False)
    replay_report.to_csv(output_dir / "replay_consistency.csv", index=False)
    anchored.anchored_events.to_csv(output_dir / "anchor_report.csv", index=False)
    anchored.simulated_events.to_csv(output_dir / "simulated_events_regrouped.csv", index=False)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            asdict(cfg)
            | {
                "initial_q": initial_q,
                "window_start": str(origin),
                "market_side": consuming_side,
                "selection_stop_seconds": cfg.selection_stop_seconds,
                "replay_consistency": replay_summary,
            },
            f,
            indent=2,
        )

    plot_path = output_dir / "conditional_queue_paths.png"
    _plot_queue_paths(
        plot_path,
        grid,
        factual_queue,
        conditioning_queue,
        mechanical_no_us_queue,
        simulated_queues,
        own_order_times,
        cfg,
    )
    dq_plot_path = output_dir / "conditional_dq_paths.png"
    _plot_dq_paths(
        dq_plot_path,
        grid,
        simulated_offsets,
        mechanical_offset,
        own_order_times,
        cfg,
    )
    divergence_plot_path = output_dir / "queue_replay_divergence.png"
    _plot_replay_divergence(
        divergence_plot_path,
        replay_report,
        origin=origin,
        queue_col=cfg.queue_col,
    )

    unit_counts = _unit_counts(
        window,
        passive_flags,
        raw_side=cfg.raw_side,
        market_side=consuming_side,
    )
    identity_error = (
        float(np.max(np.abs(simulated_queues - factual_queue[:, None])))
        if cfg.selection_policy == "none" and simulated_queues.size > 0
        else None
    )
    return {
        "output_dir": output_dir,
        "plot_path": plot_path,
        "dq_plot_path": dq_plot_path,
        "divergence_plot_path": divergence_plot_path,
        "initial_q": initial_q,
        "n_rows": len(window),
        "n_flagged_passive_limits": int(passive_flags.sum()),
        "n_background_limit_units": unit_counts["background_limit"],
        "n_background_cancel_units": unit_counts["background_cancel"],
        "n_market_units": unit_counts["market"],
        "no_passive_identity_max_abs_error": identity_error,
        "replay_max_abs_residual": replay_summary["max_abs_residual"],
        "replay_nonzero_residual_rows": replay_summary["nonzero_residual_rows"],
        "replay_final_level_diff": replay_summary["final_level_diff"],
        "replay_max_abs_level_diff": replay_summary["max_abs_level_diff"],
    }


def _unit_counts(
    window: pd.DataFrame,
    passive_flags: np.ndarray,
    *,
    raw_side: str,
    market_side: str,
) -> dict[str, int]:
    types = window["order_type"].astype(str).str.lower().to_numpy()
    sides = window["side"].astype(str).to_numpy()
    qty = window["qty"].to_numpy(dtype=np.int64)
    posting = sides == raw_side
    consuming = sides == market_side
    return {
        "background_limit": int(qty[(types == "limit") & posting & ~passive_flags].sum()),
        "background_cancel": int(qty[(types == "cancel") & posting].sum()),
        "market": int(qty[(types == "market") & consuming].sum()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-path", default=QueuePipelineConfig.parquet_path)
    parser.add_argument("--output-dir", default=QueuePipelineConfig.output_dir)
    parser.add_argument("--horizon-seconds", type=float, default=QueuePipelineConfig.horizon_seconds)
    parser.add_argument("--every-seconds", type=float, default=QueuePipelineConfig.every_seconds)
    parser.add_argument("--n-simulations", type=int, default=QueuePipelineConfig.n_simulations)
    parser.add_argument("--n-grid", type=int, default=QueuePipelineConfig.n_grid)
    parser.add_argument("--raw-side", default=QueuePipelineConfig.raw_side)
    parser.add_argument("--queue-col", default=QueuePipelineConfig.queue_col)
    parser.add_argument("--market-side", default=QueuePipelineConfig.market_side)
    parser.add_argument(
        "--selection-policy",
        choices=["first_every", "random_fraction", "indices", "none"],
        default=QueuePipelineConfig.selection_policy,
    )
    parser.add_argument("--selection-fraction", type=float, default=None)
    parser.add_argument("--selection-indices", default="")
    parser.add_argument("--selection-index-base", type=int, default=QueuePipelineConfig.selection_index_base)
    parser.add_argument("--selection-stop-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=QueuePipelineConfig.seed)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--a-l", type=float, default=QueuePipelineConfig.a_l)
    parser.add_argument("--b-l", type=float, default=QueuePipelineConfig.b_l)
    parser.add_argument("--a-c", type=float, default=QueuePipelineConfig.a_c)
    parser.add_argument("--b-c", type=float, default=QueuePipelineConfig.b_c)
    parser.add_argument("--require-replay-match", action="store_true")
    parser.add_argument("--replay-tolerance", type=float, default=QueuePipelineConfig.replay_tolerance)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = QueuePipelineConfig(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        horizon_seconds=args.horizon_seconds,
        every_seconds=args.every_seconds,
        n_simulations=args.n_simulations,
        n_grid=args.n_grid,
        raw_side=args.raw_side,
        queue_col=args.queue_col,
        market_side=args.market_side,
        selection_policy=args.selection_policy,
        selection_fraction=args.selection_fraction,
        selection_indices=_parse_indices(args.selection_indices),
        selection_index_base=args.selection_index_base,
        selection_stop_seconds=args.selection_stop_seconds,
        seed=args.seed,
        start_time=args.start_time,
        a_l=args.a_l,
        b_l=args.b_l,
        a_c=args.a_c,
        b_c=args.b_c,
        require_replay_match=args.require_replay_match,
        replay_tolerance=args.replay_tolerance,
    )
    summary = run_queue_pipeline(cfg)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2))


def _parse_indices(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if not raw:
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
