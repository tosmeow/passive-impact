"""Filtered passive execution impact-cost pipeline."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.anchored_simulator import simulate_anchored_queue_paths
from ...core.cost_utils import event_seconds
from ...core.latency_filters import LatencyFilterConfig, select_latency_orders
from ...core.level_execution import market_side_for_queue
from ...core.passive_impact import (
    PassiveImpactModelConfig,
    execution_cost_jump_series,
    passive_cost_from_fills,
    passive_impact_path_from_queue_samples,
    validate_passive_impact_model_config,
)
from ...core.reduced_form_impact import (
    DEFAULT_PROPAGATOR_BETA,
    DEFAULT_PROPAGATOR_GAMMA,
    DEFAULT_PROPAGATOR_KAPPA,
    DEFAULT_PROPAGATOR_WEIGHTS,
)
from .queue_pipeline import infer_initial_queue


@dataclass(frozen=True)
class ImpactCostPipelineConfig:
    """Configuration for the filtered passive impact-cost experiment.

    The pipeline consumes selected orders/fills from `execution_latency_grid`
    plus the processed depth input, re-simulates anchored no-us queues for each
    selected window, evaluates the passive flow-impact path on market times,
    and aggregates fill-level cost contributions.
    """

    aggregated_path: str = "experiments/impact_cost/load_experiments/data/processed/factual_2025_05_29_esm5.parquet"
    latency_orders_path: str = (
        "experiments/impact_cost/archive/diagnostics/data/execution_latency_grid/"
        "passive_execution_latencies_by_minute.csv"
    )
    latency_fills_path: str = (
        "experiments/impact_cost/archive/diagnostics/data/execution_latency_grid/"
        "passive_execution_fills_by_minute.csv"
    )
    output_dir: str = "experiments/impact_cost/archive/diagnostics/data/impact_cost_latency_le_30s"
    max_latency_seconds: float | None = 30.0
    selection_mode: str = "orders"
    min_orders: int = 1
    required_slots: tuple[int, ...] = ()
    max_windows: int | None = None
    horizon_seconds: float = 30.0
    n_simulations: int = 25
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    seed: int = 2027
    a_l: float = 184.1372
    b_l: float = -0.0097
    a_c: float = 184.0456
    b_c: float = 0.00989
    impact_model: str = "reduced_form"
    propagator_kappa: float = DEFAULT_PROPAGATOR_KAPPA
    propagator_gamma: float = DEFAULT_PROPAGATOR_GAMMA
    propagator_weights: tuple[float, ...] = DEFAULT_PROPAGATOR_WEIGHTS
    propagator_beta: tuple[float, ...] = DEFAULT_PROPAGATOR_BETA
    propagator_tail_zeta: float = 0.0
    c_kappa: float = -2.1766e-6
    hawkes_mu: float = 1.0
    hawkes_alpha: tuple[float, ...] = (0.065, 0.2, 0.325, 0.65)
    hawkes_beta: tuple[float, ...] = (0.15, 0.60, 2.5, 10.0)


def run_impact_cost_pipeline(cfg: ImpactCostPipelineConfig) -> dict[str, Any]:
    """Run the selected-window impact-cost pipeline and return output summary."""
    _validate_config(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(cfg.latency_orders_path)
    fills = pd.read_csv(cfg.latency_fills_path)
    selected_orders, selected_windows = select_latency_orders(
        orders,
        LatencyFilterConfig(
            max_latency_seconds=cfg.max_latency_seconds,
            require_completed=True,
            selection_mode=cfg.selection_mode,
            min_orders=cfg.min_orders,
            required_slots=cfg.required_slots,
        ),
    )
    if cfg.max_windows is not None:
        selected_windows = selected_windows.head(int(cfg.max_windows)).copy()
        keep_window_ids = set(selected_windows["window_id"].astype(int))
        selected_orders = selected_orders[
            selected_orders["window_id"].astype(int).isin(keep_window_ids)
        ].copy()

    selected_fills = fills[fills["order_id"].isin(selected_orders["order_id"])].copy()
    selected_orders.to_csv(output_dir / "selected_latency_orders.csv", index=False)
    selected_windows.to_csv(output_dir / "selected_latency_windows.csv", index=False)
    selected_fills.to_csv(output_dir / "selected_latency_fills.csv", index=False)

    aggregated = _load_aggregated_depth(cfg.aggregated_path)
    simproj = _import_simproj()

    cost_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    for window_row in selected_windows.itertuples(index=False):
        window_id = int(window_row.window_id)
        window_start = pd.Timestamp(window_row.window_start)
        window_orders = selected_orders[selected_orders["window_id"].astype(int) == window_id]
        window_fills = selected_fills[selected_fills["window_id"].astype(int) == window_id]
        if window_orders.empty:
            continue
        window_costs, window_fill_costs = _run_one_window(
            cfg,
            simproj,
            aggregated,
            window_id=window_id,
            window_start=window_start,
            selected_orders=window_orders,
            selected_fills=window_fills,
        )
        cost_rows.extend(window_costs)
        fill_rows.extend(window_fill_costs)

    cost_samples = pd.DataFrame(cost_rows)
    fill_costs = pd.DataFrame(fill_rows)
    window_summary = _summarize_costs(cost_samples, selected_windows)

    cost_samples.to_csv(output_dir / "impact_cost_samples.csv", index=False)
    fill_costs.to_csv(output_dir / "impact_cost_fill_contributions.csv", index=False)
    window_summary.to_csv(output_dir / "impact_cost_window_summary.csv", index=False)
    _plot_cost_distribution(output_dir / "impact_cost_distribution.png", cost_samples)

    with open(output_dir / "impact_cost_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return {
        "output_dir": output_dir,
        "n_selected_windows": int(len(selected_windows)),
        "n_selected_orders": int(len(selected_orders)),
        "n_selected_fills": int(len(selected_fills)),
        "n_cost_samples": int(len(cost_samples)),
        "mean_cost": float(cost_samples["cost"].mean()) if len(cost_samples) else float("nan"),
        "median_cost": float(cost_samples["cost"].median()) if len(cost_samples) else float("nan"),
        "cost_samples_path": output_dir / "impact_cost_samples.csv",
        "window_summary_path": output_dir / "impact_cost_window_summary.csv",
        "fill_contributions_path": output_dir / "impact_cost_fill_contributions.csv",
        "cost_plot_path": output_dir / "impact_cost_distribution.png",
    }


def _run_one_window(
    cfg: ImpactCostPipelineConfig,
    simproj: Any,
    aggregated: pd.DataFrame,
    *,
    window_id: int,
    window_start: pd.Timestamp,
    selected_orders: pd.DataFrame,
    selected_fills: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    window = _window_from_aggregated(aggregated, window_start, cfg.horizon_seconds)
    if window.empty:
        raise ValueError(f"no aggregated rows for window {window_start}")

    initial_q = infer_initial_queue(
        window,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
    )
    passive_source_rows = set(selected_orders["source_row_pos"].astype(int))
    passive_flags = window["source_row_pos"].astype(int).isin(passive_source_rows).to_numpy()

    consuming_side = market_side_for_queue(
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
    )
    market_times = _market_times(window, origin=window_start, market_side=consuming_side)
    selected_fills = selected_fills[
        selected_fills["fill_time_s"].to_numpy(dtype=np.float64) <= cfg.horizon_seconds
    ].copy()
    selected_qty = int(selected_orders["post_qty"].sum())
    filled_qty = int(selected_fills["qty"].sum()) if not selected_fills.empty else 0

    if market_times.size == 0:
        return _zero_cost_rows(cfg, window_id, window_start, selected_orders, selected_qty, filled_qty), []

    anchored = simulate_anchored_queue_paths(
        window,
        passive_flags,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=consuming_side,
        initial_q=initial_q,
        horizon_seconds=cfg.horizon_seconds,
        grid=market_times,
        n_simulations=cfg.n_simulations,
        seed=cfg.seed + window_id,
        a_l=cfg.a_l,
        b_l=cfg.b_l,
        a_c=cfg.a_c,
        b_c=cfg.b_c,
        origin=window_start,
    )
    q_bar_samples = _queue_to_u32(anchored.factual_queue)
    impact_cfg = _passive_impact_model_config(cfg)

    cost_rows = []
    fill_cost_rows = []
    for sim_idx in range(cfg.n_simulations):
        q_samples = _queue_to_u32(anchored.simulated_queues[:, sim_idx])
        impact = passive_impact_path_from_queue_samples(
            q_samples,
            q_bar_samples,
            market_times,
            queue_col=cfg.queue_col,
            cfg=impact_cfg,
            simproj=simproj,
        )
        cost = passive_cost_from_fills(
            selected_fills,
            market_times=market_times,
            impact=impact,
            time_col="fill_time_s",
            qty_col="qty",
        )
        cost_rows.append(
            {
                "window_id": window_id,
                "window_start": str(window_start),
                "simulation": sim_idx,
                "cost": cost,
                "n_selected_orders": int(len(selected_orders)),
                "selected_qty": selected_qty,
                "filled_qty": filled_qty,
                "n_market_events": int(market_times.size),
                "max_latency_s": float(selected_orders["latency_s"].max()),
                "mean_latency_s": float(selected_orders["latency_s"].mean()),
            }
        )
        jumps = execution_cost_jump_series(
            selected_fills,
            market_times=market_times,
            impact=impact,
            time_col="fill_time_s",
            qty_col="qty",
            copy_columns=("order_id", "order_slot"),
            extra_columns={
                "window_id": window_id,
                "window_start": str(window_start),
                "simulation": sim_idx,
            },
        )
        fill_cost_rows.extend(jumps.to_dict("records"))
    return cost_rows, fill_cost_rows


def _zero_cost_rows(
    cfg: ImpactCostPipelineConfig,
    window_id: int,
    window_start: pd.Timestamp,
    selected_orders: pd.DataFrame,
    selected_qty: int,
    filled_qty: int,
) -> list[dict[str, Any]]:
    return [
        {
            "window_id": window_id,
            "window_start": str(window_start),
            "simulation": sim_idx,
            "cost": 0.0,
            "n_selected_orders": int(len(selected_orders)),
            "selected_qty": selected_qty,
            "filled_qty": filled_qty,
            "n_market_events": 0,
            "max_latency_s": float(selected_orders["latency_s"].max()),
            "mean_latency_s": float(selected_orders["latency_s"].mean()),
        }
        for sim_idx in range(cfg.n_simulations)
    ]


def _load_aggregated_depth(path: str) -> pd.DataFrame:
    columns = ["ts", "order_type", "side", "qty", "q_b", "q_a"]
    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, usecols=columns)
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df = pd.read_parquet(input_path, columns=columns)
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["source_row_pos"] = np.arange(len(df), dtype=np.int64)
    return df


def _window_from_aggregated(
    aggregated: pd.DataFrame,
    start: pd.Timestamp,
    horizon_seconds: float,
) -> pd.DataFrame:
    ts = aggregated["ts"]
    if getattr(ts.dt, "tz", None) is not None and start.tzinfo is None:
        start = start.tz_localize(ts.dt.tz)
    end = start + pd.Timedelta(seconds=float(horizon_seconds))
    return aggregated[(ts >= start) & (ts <= end)].copy().reset_index(drop=True)


def _market_times(
    window: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    market_side: str,
) -> np.ndarray:
    types = window["order_type"].astype(str).str.lower()
    sides = window["side"].astype(str)
    market_rows = window[(types == "market") & (sides == market_side)]
    if market_rows.empty:
        return np.array([], dtype=np.float64)
    return event_seconds(market_rows, origin=origin).astype(np.float64)


def _queue_to_u32(values: np.ndarray) -> np.ndarray:
    return np.rint(np.maximum(values, 0.0)).astype(np.uint32)


def _validate_config(cfg: ImpactCostPipelineConfig) -> None:
    validate_passive_impact_model_config(_passive_impact_model_config(cfg))
    if len(cfg.propagator_weights) != len(cfg.propagator_beta):
        raise ValueError("propagator_weights and propagator_beta must have matching lengths")
    if any(float(beta) <= 0.0 for beta in cfg.propagator_beta):
        raise ValueError("propagator_beta values must be positive")


def _passive_impact_model_config(cfg: ImpactCostPipelineConfig) -> PassiveImpactModelConfig:
    return PassiveImpactModelConfig(
        impact_model=cfg.impact_model,
        propagator_kappa=cfg.propagator_kappa,
        propagator_gamma=cfg.propagator_gamma,
        propagator_weights=cfg.propagator_weights,
        propagator_beta=cfg.propagator_beta,
        propagator_tail_zeta=cfg.propagator_tail_zeta,
        c_kappa=cfg.c_kappa,
        hawkes_mu=cfg.hawkes_mu,
        hawkes_alpha=cfg.hawkes_alpha,
        hawkes_beta=cfg.hawkes_beta,
        b_l=cfg.b_l,
        b_c=cfg.b_c,
    )


def _summarize_costs(cost_samples: pd.DataFrame, selected_windows: pd.DataFrame) -> pd.DataFrame:
    base = selected_windows.copy()
    if cost_samples.empty:
        return base
    grouped = cost_samples.groupby("window_id", sort=True)
    summary = grouped.agg(
        n_cost_samples=("cost", "count"),
        mean_cost=("cost", "mean"),
        median_cost=("cost", "median"),
        std_cost=("cost", "std"),
        q05_cost=("cost", lambda x: float(np.quantile(x, 0.05))),
        q95_cost=("cost", lambda x: float(np.quantile(x, 0.95))),
        n_market_events=("n_market_events", "first"),
        filled_qty=("filled_qty", "first"),
    ).reset_index()
    return base.merge(summary, on="window_id", how="left")


def _plot_cost_distribution(path: Path, cost_samples: pd.DataFrame) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.4))
    if not cost_samples.empty:
        values = cost_samples["cost"].to_numpy(dtype=np.float64)
        ax.hist(values, bins=45, color="#557aa6", alpha=0.82)
        ax.axvline(float(np.mean(values)), color="#b83232", linewidth=1.4, label=f"mean {np.mean(values):.3g}")
        ax.axvline(float(np.median(values)), color="#1f7a3d", linewidth=1.4, label=f"median {np.median(values):.3g}")
    ax.set_title("Passive execution impact-cost distribution")
    ax.set_xlabel("signed cost")
    ax.set_ylabel("simulation-window count")
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
    parser.add_argument("--aggregated-path", default=ImpactCostPipelineConfig.aggregated_path)
    parser.add_argument("--latency-orders-path", default=ImpactCostPipelineConfig.latency_orders_path)
    parser.add_argument("--latency-fills-path", default=ImpactCostPipelineConfig.latency_fills_path)
    parser.add_argument("--output-dir", default=ImpactCostPipelineConfig.output_dir)
    parser.add_argument("--max-latency-seconds", type=float, default=ImpactCostPipelineConfig.max_latency_seconds)
    parser.add_argument(
        "--selection-mode",
        choices=["orders", "window_any", "window_at_least", "window_all"],
        default=ImpactCostPipelineConfig.selection_mode,
    )
    parser.add_argument("--min-orders", type=int, default=ImpactCostPipelineConfig.min_orders)
    parser.add_argument("--required-slots", default="")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--horizon-seconds", type=float, default=ImpactCostPipelineConfig.horizon_seconds)
    parser.add_argument("--n-simulations", type=int, default=ImpactCostPipelineConfig.n_simulations)
    parser.add_argument("--raw-side", default=ImpactCostPipelineConfig.raw_side)
    parser.add_argument("--queue-col", default=ImpactCostPipelineConfig.queue_col)
    parser.add_argument("--market-side", default=ImpactCostPipelineConfig.market_side)
    parser.add_argument("--seed", type=int, default=ImpactCostPipelineConfig.seed)
    parser.add_argument("--a-l", type=float, default=ImpactCostPipelineConfig.a_l)
    parser.add_argument("--b-l", type=float, default=ImpactCostPipelineConfig.b_l)
    parser.add_argument("--a-c", type=float, default=ImpactCostPipelineConfig.a_c)
    parser.add_argument("--b-c", type=float, default=ImpactCostPipelineConfig.b_c)
    parser.add_argument(
        "--impact-model",
        choices=["reduced_form", "tail_propagator", "propagator_tail", "structural"],
        default=ImpactCostPipelineConfig.impact_model,
    )
    parser.add_argument("--propagator-kappa", type=float, default=ImpactCostPipelineConfig.propagator_kappa)
    parser.add_argument("--propagator-gamma", type=float, default=ImpactCostPipelineConfig.propagator_gamma)
    parser.add_argument(
        "--propagator-weights",
        default=",".join(str(x) for x in ImpactCostPipelineConfig.propagator_weights),
    )
    parser.add_argument(
        "--propagator-beta",
        default=",".join(str(x) for x in ImpactCostPipelineConfig.propagator_beta),
    )
    parser.add_argument("--propagator-tail-zeta", type=float, default=ImpactCostPipelineConfig.propagator_tail_zeta)
    parser.add_argument("--c-kappa", type=float, default=ImpactCostPipelineConfig.c_kappa)
    parser.add_argument("--hawkes-mu", type=float, default=ImpactCostPipelineConfig.hawkes_mu)
    parser.add_argument("--hawkes-alpha", default=",".join(str(x) for x in ImpactCostPipelineConfig.hawkes_alpha))
    parser.add_argument("--hawkes-beta", default=",".join(str(x) for x in ImpactCostPipelineConfig.hawkes_beta))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ImpactCostPipelineConfig(
        aggregated_path=args.aggregated_path,
        latency_orders_path=args.latency_orders_path,
        latency_fills_path=args.latency_fills_path,
        output_dir=args.output_dir,
        max_latency_seconds=args.max_latency_seconds,
        selection_mode=args.selection_mode,
        min_orders=args.min_orders,
        required_slots=_parse_ints(args.required_slots),
        max_windows=args.max_windows,
        horizon_seconds=args.horizon_seconds,
        n_simulations=args.n_simulations,
        raw_side=args.raw_side,
        queue_col=args.queue_col,
        market_side=args.market_side,
        seed=args.seed,
        a_l=args.a_l,
        b_l=args.b_l,
        a_c=args.a_c,
        b_c=args.b_c,
        impact_model=args.impact_model,
        propagator_kappa=args.propagator_kappa,
        propagator_gamma=args.propagator_gamma,
        propagator_weights=_parse_floats(args.propagator_weights),
        propagator_beta=_parse_floats(args.propagator_beta),
        propagator_tail_zeta=args.propagator_tail_zeta,
        c_kappa=args.c_kappa,
        hawkes_mu=args.hawkes_mu,
        hawkes_alpha=_parse_floats(args.hawkes_alpha),
        hawkes_beta=_parse_floats(args.hawkes_beta),
    )
    summary = run_impact_cost_pipeline(cfg)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2))


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
