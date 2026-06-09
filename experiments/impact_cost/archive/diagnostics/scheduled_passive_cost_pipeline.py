"""Scheduled passive posting/fill impact-cost path experiment."""
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
from ...core.cost_utils import LIMIT, MARKET, event_seconds
from ...core.level_execution import market_side_for_queue
from ...core.passive_impact import (
    PassiveImpactModelConfig,
    execution_cost_jump_series,
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
class ScheduledPassiveCostConfig:
    """Configuration for a scheduled passive posting/fill validation run.

    For each candidate episode, the pipeline selects the first factual passive
    limit row in each posting bucket, simulates anchored no-us queues, then
    applies a synthetic fill schedule to produce cumulative impact-cost paths.
    """

    aggregated_path: str = "experiments/impact_cost/load_experiments/data/processed/factual_2025_05_29_esm5.parquet"
    output_dir: str = "experiments/impact_cost/archive/diagnostics/data/scheduled_passive_cost"
    episode_spacing_seconds: float = 60.0
    max_episodes: int | None = None
    randomize_episodes: bool = False
    posting_spacing_seconds: float = 0.010
    n_posting_slots: int = 10
    fill_start_seconds: float = 0.150
    fill_schedule: str = "exponential_quantile"
    fill_spacing_seconds: float = 0.002
    fill_half_life_seconds: float = 0.010
    n_filled_orders: int | None = None
    horizon_seconds: float = 2.0
    output_step_seconds: float = 0.002
    n_simulations: int = 25
    require_full_posting_grid: bool = True
    min_market_events: int = 1
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    start_time: str | None = None
    end_time: str | None = None
    seed: int = 2027
    a_l: float = 184.1372
    b_l: float = -0.000097
    a_c: float = 184.0456
    b_c: float = 0.0000989
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


def run_scheduled_passive_cost_pipeline(cfg: ScheduledPassiveCostConfig) -> dict[str, Any]:
    """Run scheduled passive cost paths and write CSV/plot outputs."""
    _validate_config(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = _load_aggregated_depth(cfg.aggregated_path)
    candidates = _candidate_episodes(aggregated, cfg)
    simproj = _import_simproj()
    output_grid = _output_grid(cfg)
    impact_cfg = _passive_impact_model_config(cfg)

    episode_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    jump_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    impact_path_rows: list[dict[str, Any]] = []

    for candidate in candidates.itertuples(index=False):
        episode_id = int(candidate.episode_id)
        window_start = pd.Timestamp(candidate.window_start)
        window = _window_from_aggregated(aggregated, window_start, cfg.horizon_seconds)
        if window.empty:
            episode_rows.append(_episode_status(candidate, status="empty_window"))
            continue

        consuming_side = market_side_for_queue(
            raw_side=cfg.raw_side,
            queue_col=cfg.queue_col,
            market_side=cfg.market_side,
        )
        passive_flags = _select_posting_grid_limits(window, cfg, origin=window_start)
        selected_orders = _selected_orders(window, passive_flags, episode_id, origin=window_start)
        n_posted = int(len(selected_orders))
        expected_posts = int(cfg.n_posting_slots)
        if cfg.require_full_posting_grid and n_posted != expected_posts:
            episode_rows.append(
                _episode_status(
                    candidate,
                    status="incomplete_posting_grid",
                    n_posted_orders=n_posted,
                )
            )
            continue

        market_times = _market_times(window, origin=window_start, market_side=consuming_side)
        if market_times.size < cfg.min_market_events:
            episode_rows.append(
                _episode_status(
                    candidate,
                    status="too_few_market_events",
                    n_posted_orders=n_posted,
                    n_market_events=int(market_times.size),
                )
            )
            continue

        fills = _synthetic_fills(selected_orders, cfg)
        if fills.empty:
            episode_rows.append(
                _episode_status(
                    candidate,
                    status="no_synthetic_fills",
                    n_posted_orders=n_posted,
                    n_market_events=int(market_times.size),
                )
            )
            continue

        initial_q = infer_initial_queue(
            window,
            raw_side=cfg.raw_side,
            queue_col=cfg.queue_col,
            market_side=consuming_side,
        )
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
            seed=cfg.seed + episode_id,
            a_l=cfg.a_l,
            b_l=cfg.b_l,
            a_c=cfg.a_c,
            b_c=cfg.b_c,
            origin=window_start,
        )
        q_bar_samples = _queue_to_u32(anchored.factual_queue)

        order_rows.extend(selected_orders.to_dict("records"))
        fill_rows.extend(fills.to_dict("records"))
        final_costs: list[float] = []
        for sim_idx in range(cfg.n_simulations):
            q_samples = _queue_to_u32(anchored.simulated_queues[:, sim_idx])
            impact_at_markets = passive_impact_path_from_queue_samples(
                q_samples,
                q_bar_samples,
                market_times,
                queue_col=cfg.queue_col,
                cfg=impact_cfg,
                simproj=simproj,
            )
            impact_on_grid = _sample_previous_value(
                event_times=market_times,
                event_values=impact_at_markets,
                output_grid=output_grid,
                initial_value=0.0,
            )
            jumps = execution_cost_jump_series(
                fills,
                market_times=market_times,
                impact=impact_at_markets,
                time_col="fill_time_s",
                qty_col="qty",
                copy_columns=("order_id", "order_slot", "post_time_s"),
                extra_columns={
                    "episode_id": episode_id,
                    "simulation": sim_idx,
                    "window_start": str(window_start),
                    "n_posted_orders": n_posted,
                    "n_filled_orders": int(len(fills)),
                },
            )
            jump_rows.extend(jumps.to_dict("records"))
            cumulative_path = _sample_previous_value(
                event_times=jumps["fill_time_s"].to_numpy(dtype=np.float64),
                event_values=jumps["cumulative_cost"].to_numpy(dtype=np.float64),
                output_grid=output_grid,
                initial_value=0.0,
            )
            final_cost = float(cumulative_path[-1]) if cumulative_path.size else 0.0
            final_costs.append(final_cost)
            path_rows.extend(
                {
                    "episode_id": episode_id,
                    "simulation": sim_idx,
                    "window_start": str(window_start),
                    "time_s": float(t),
                    "cumulative_cost": float(value),
                    "n_posted_orders": n_posted,
                    "n_filled_orders": int(len(fills)),
                }
                for t, value in zip(output_grid, cumulative_path)
            )
            impact_path_rows.extend(
                {
                    "episode_id": episode_id,
                    "simulation": sim_idx,
                    "window_start": str(window_start),
                    "time_s": float(t),
                    "price_impact": float(value),
                    "n_posted_orders": n_posted,
                    "n_filled_orders": int(len(fills)),
                }
                for t, value in zip(output_grid, impact_on_grid)
            )

        episode_rows.append(
            _episode_status(
                candidate,
                status="ok",
                n_posted_orders=n_posted,
                n_filled_orders=int(len(fills)),
                n_market_events=int(market_times.size),
                selected_qty=int(selected_orders["qty"].sum()),
                filled_qty=int(fills["qty"].sum()),
                initial_q=int(initial_q),
                mean_final_cost=float(np.mean(final_costs)) if final_costs else 0.0,
                median_final_cost=float(np.median(final_costs)) if final_costs else 0.0,
            )
        )

    episodes = pd.DataFrame(episode_rows)
    orders = pd.DataFrame(order_rows)
    fills = pd.DataFrame(fill_rows)
    jumps = pd.DataFrame(jump_rows)
    samples = pd.DataFrame(path_rows)
    impact_samples = pd.DataFrame(impact_path_rows)
    summary = _summarize_cost_paths(samples, output_grid)
    impact_summary = _summarize_price_impact_paths(impact_samples, output_grid)

    episodes.to_csv(output_dir / "episode_summary.csv", index=False)
    orders.to_csv(output_dir / "selected_orders.csv", index=False)
    fills.to_csv(output_dir / "synthetic_fills.csv", index=False)
    jumps.to_csv(output_dir / "impact_cost_fill_jumps.csv", index=False)
    samples.to_csv(output_dir / "impact_cost_path_samples.csv", index=False)
    summary.to_csv(output_dir / "impact_cost_path_summary.csv", index=False)
    impact_samples.to_csv(output_dir / "price_impact_path_samples.csv", index=False)
    impact_summary.to_csv(output_dir / "price_impact_path_summary.csv", index=False)
    _plot_cost_paths(
        output_dir / "impact_cost_paths.png",
        samples,
        summary,
        impact_summary,
        cfg,
    )

    with open(output_dir / "scheduled_passive_cost_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    ok = episodes[episodes["status"] == "ok"] if not episodes.empty else episodes
    return {
        "output_dir": output_dir,
        "n_candidate_episodes": int(len(candidates)),
        "n_ok_episodes": int(len(ok)),
        "n_path_samples": int(len(samples)),
        "n_fill_jumps": int(len(jumps)),
        "mean_final_cost": float(ok["mean_final_cost"].mean()) if len(ok) else float("nan"),
        "median_episode_final_cost": float(ok["median_final_cost"].median()) if len(ok) else float("nan"),
        "path_samples_path": output_dir / "impact_cost_path_samples.csv",
        "path_summary_path": output_dir / "impact_cost_path_summary.csv",
        "price_impact_samples_path": output_dir / "price_impact_path_samples.csv",
        "price_impact_summary_path": output_dir / "price_impact_path_summary.csv",
        "fill_jumps_path": output_dir / "impact_cost_fill_jumps.csv",
        "plot_path": output_dir / "impact_cost_paths.png",
    }


def _candidate_episodes(
    aggregated: pd.DataFrame,
    cfg: ScheduledPassiveCostConfig,
) -> pd.DataFrame:
    types = aggregated["order_type"].astype(str).str.lower()
    sides = aggregated["side"].astype(str)
    limit_rows = aggregated[(types == LIMIT) & (sides == cfg.raw_side)][
        ["source_row_pos", "ts", "qty"]
    ].copy()
    if limit_rows.empty:
        return pd.DataFrame(
            {
                "episode_id": pd.Series(dtype=np.int64),
                "window_start": pd.Series(dtype=object),
                "first_source_row_pos": pd.Series(dtype=np.int64),
            }
        )

    origin = aggregated["ts"].iloc[0]
    limit_seconds = event_seconds(limit_rows, origin=origin)
    max_start_s = float((aggregated["ts"].iloc[-1] - origin).total_seconds()) - float(
        cfg.horizon_seconds
    )
    keep = limit_seconds <= max_start_s
    if cfg.start_time is not None:
        start_bound = _timestamp_like(aggregated["ts"], cfg.start_time)
        keep &= (limit_rows["ts"] >= start_bound).to_numpy()
    if cfg.end_time is not None:
        end_bound = _timestamp_like(aggregated["ts"], cfg.end_time)
        keep &= (limit_rows["ts"] <= end_bound).to_numpy()
    limit_rows = limit_rows[keep].copy()
    limit_seconds = limit_seconds[keep]
    if limit_rows.empty:
        return pd.DataFrame(
            {
                "episode_id": pd.Series(dtype=np.int64),
                "window_start": pd.Series(dtype=object),
                "first_source_row_pos": pd.Series(dtype=np.int64),
            }
        )

    if cfg.episode_spacing_seconds > 0.0:
        spacing = float(cfg.episode_spacing_seconds)
        start_s = float(limit_seconds[0])
        buckets = np.floor((limit_seconds - start_s) / spacing).astype(np.int64)
        _, first_idx = np.unique(buckets, return_index=True)
        selected = limit_rows.iloc[np.sort(first_idx)].copy()
    else:
        selected = limit_rows.copy()

    if cfg.max_episodes is not None and len(selected) > int(cfg.max_episodes):
        if cfg.randomize_episodes:
            rng = np.random.default_rng(cfg.seed)
            idx = np.sort(rng.choice(np.arange(len(selected)), size=int(cfg.max_episodes), replace=False))
            selected = selected.iloc[idx].copy()
        else:
            selected = selected.head(int(cfg.max_episodes)).copy()

    selected = selected.reset_index(drop=True)
    return pd.DataFrame(
        {
            "episode_id": np.arange(len(selected), dtype=np.int64),
            "window_start": selected["ts"].astype(str).to_numpy(),
            "first_source_row_pos": selected["source_row_pos"].astype(int).to_numpy(),
        }
    )


def _select_posting_grid_limits(
    window: pd.DataFrame,
    cfg: ScheduledPassiveCostConfig,
    *,
    origin: pd.Timestamp,
) -> np.ndarray:
    seconds = event_seconds(window, origin=origin)
    types = window["order_type"].astype(str).str.lower().to_numpy()
    sides = window["side"].astype(str).to_numpy()
    in_posting_window = (
        (seconds >= 0.0)
        & (seconds < float(cfg.n_posting_slots) * float(cfg.posting_spacing_seconds))
    )
    is_limit = (types == LIMIT) & (sides == cfg.raw_side) & in_posting_window
    candidate_pos = np.flatnonzero(is_limit)
    flags = np.zeros(len(window), dtype=bool)
    if candidate_pos.size == 0:
        return flags

    buckets = np.floor(seconds[candidate_pos] / float(cfg.posting_spacing_seconds)).astype(np.int64)
    valid = (buckets >= 0) & (buckets < int(cfg.n_posting_slots))
    candidate_pos = candidate_pos[valid]
    buckets = buckets[valid]
    if candidate_pos.size == 0:
        return flags

    _, first_idx = np.unique(buckets, return_index=True)
    flags[candidate_pos[np.sort(first_idx)]] = True
    return flags


def _selected_orders(
    window: pd.DataFrame,
    passive_flags: np.ndarray,
    episode_id: int,
    *,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    selected = window[passive_flags].copy()
    if selected.empty:
        return pd.DataFrame(
            {
                "episode_id": pd.Series(dtype=np.int64),
                "order_id": pd.Series(dtype=np.int64),
                "order_slot": pd.Series(dtype=np.int64),
                "source_row_pos": pd.Series(dtype=np.int64),
                "post_ts": pd.Series(dtype=object),
                "post_time_s": pd.Series(dtype=np.float64),
                "qty": pd.Series(dtype=np.int64),
            }
        )
    post_times = event_seconds(selected, origin=origin)
    return pd.DataFrame(
        {
            "episode_id": int(episode_id),
            "order_id": np.arange(len(selected), dtype=np.int64),
            "order_slot": np.arange(len(selected), dtype=np.int64),
            "source_row_pos": selected["source_row_pos"].astype(int).to_numpy(),
            "post_ts": selected["ts"].astype(str).to_numpy(),
            "post_time_s": post_times.astype(np.float64),
            "qty": selected["qty"].astype(int).to_numpy(),
        }
    )


def _synthetic_fills(
    selected_orders: pd.DataFrame,
    cfg: ScheduledPassiveCostConfig,
) -> pd.DataFrame:
    n_fills = len(selected_orders) if cfg.n_filled_orders is None else min(
        int(cfg.n_filled_orders),
        len(selected_orders),
    )
    if n_fills <= 0:
        return pd.DataFrame(
            {
                "episode_id": pd.Series(dtype=np.int64),
                "order_id": pd.Series(dtype=np.int64),
                "order_slot": pd.Series(dtype=np.int64),
                "post_time_s": pd.Series(dtype=np.float64),
                "fill_time_s": pd.Series(dtype=np.float64),
                "qty": pd.Series(dtype=np.int64),
            }
        )
    fills = selected_orders.sort_values("order_slot").head(n_fills).copy()
    fill_times = _scheduled_fill_times(n_fills, cfg)
    fills = fills[fill_times <= float(cfg.horizon_seconds)].copy()
    fill_times = fill_times[: len(fills)]
    return pd.DataFrame(
        {
            "episode_id": fills["episode_id"].astype(int).to_numpy(),
            "order_id": fills["order_id"].astype(int).to_numpy(),
            "order_slot": fills["order_slot"].astype(int).to_numpy(),
            "post_time_s": fills["post_time_s"].astype(float).to_numpy(),
            "fill_time_s": fill_times,
            "qty": fills["qty"].astype(int).to_numpy(),
        }
    )


def _summarize_cost_paths(samples: pd.DataFrame, output_grid: np.ndarray) -> pd.DataFrame:
    columns = [
        "n_filled_orders",
        "time_s",
        "n_samples",
        "mean_cost",
        "std_cost",
        "q05_cost",
        "q25_cost",
        "median_cost",
        "q75_cost",
        "q95_cost",
    ]
    if samples.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})

    grouped = samples.groupby(["n_filled_orders", "time_s"], sort=True)["cumulative_cost"]
    summary = grouped.agg(
        n_samples="count",
        mean_cost="mean",
        std_cost="std",
        q05_cost=lambda x: float(np.quantile(x, 0.05)),
        q25_cost=lambda x: float(np.quantile(x, 0.25)),
        median_cost="median",
        q75_cost=lambda x: float(np.quantile(x, 0.75)),
        q95_cost=lambda x: float(np.quantile(x, 0.95)),
    ).reset_index()
    return summary[columns]


def _summarize_price_impact_paths(
    samples: pd.DataFrame,
    output_grid: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "n_filled_orders",
        "time_s",
        "n_samples",
        "mean_price_impact",
        "std_price_impact",
        "q05_price_impact",
        "q95_price_impact",
    ]
    if samples.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})

    grouped = samples.groupby(["n_filled_orders", "time_s"], sort=True)["price_impact"]
    summary = grouped.agg(
        n_samples="count",
        mean_price_impact="mean",
        std_price_impact="std",
        q05_price_impact=lambda x: float(np.quantile(x, 0.05)),
        q95_price_impact=lambda x: float(np.quantile(x, 0.95)),
    ).reset_index()
    return summary[columns]


def _plot_cost_paths(
    path: Path,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    impact_summary: pd.DataFrame,
    cfg: ScheduledPassiveCostConfig,
) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax_impact = ax.twinx()
    ax_impact.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)

    plot_samples = samples
    plot_summary = summary
    plot_impact_summary = impact_summary
    filled_label = None
    if not samples.empty:
        filled_counts = samples["n_filled_orders"].value_counts().sort_index()
        filled_label = int(filled_counts.index[-1])
        plot_samples = samples[samples["n_filled_orders"] == filled_label]
        plot_summary = summary[summary["n_filled_orders"] == filled_label]
        plot_impact_summary = impact_summary[
            impact_summary["n_filled_orders"] == filled_label
        ]

    if not plot_impact_summary.empty and plot_impact_summary["mean_price_impact"].notna().any():
        x_imp = plot_impact_summary["time_s"].to_numpy(dtype=np.float64)
        mean_imp = plot_impact_summary["mean_price_impact"].to_numpy(dtype=np.float64)
        q05_imp = plot_impact_summary["q05_price_impact"].to_numpy(dtype=np.float64)
        q95_imp = plot_impact_summary["q95_price_impact"].to_numpy(dtype=np.float64)
        ax_impact.fill_between(
            x_imp,
            q05_imp,
            q95_imp,
            color="#7a8c99",
            alpha=0.08,
            linewidth=0,
            label="price impact 5-95%",
            zorder=0,
        )
        ax_impact.plot(
            x_imp,
            mean_imp,
            color="#5d6872",
            linewidth=1.8,
            alpha=0.55,
            linestyle="--",
            label="mean price impact",
            zorder=0,
        )

    if not samples.empty:
        for _, path_rows in plot_samples.groupby(["episode_id", "simulation"], sort=False):
            path_rows = path_rows.sort_values("time_s")
            ax.step(
                path_rows["time_s"],
                path_rows["cumulative_cost"],
                where="post",
                color="0.38",
                linewidth=0.55,
                alpha=0.10,
                zorder=1,
            )

    if not plot_summary.empty and plot_summary["mean_cost"].notna().any():
        x = plot_summary["time_s"].to_numpy(dtype=np.float64)
        mean = plot_summary["mean_cost"].to_numpy(dtype=np.float64)
        q05 = plot_summary["q05_cost"].to_numpy(dtype=np.float64)
        q95 = plot_summary["q95_cost"].to_numpy(dtype=np.float64)
        ax.fill_between(
            x,
            q05,
            q95,
            step="post",
            color="#5b8cc0",
            alpha=0.20,
            linewidth=0,
            label="5-95%",
            zorder=2,
        )
        ax.step(x, mean, where="post", color="#1f5d99", linewidth=2.0, label="mean", zorder=3)

    plot_start = _plot_start_seconds(cfg)
    plot_end = _plot_end_seconds(cfg)
    posting_end = float(cfg.n_posting_slots) * float(cfg.posting_spacing_seconds)
    if plot_start < posting_end:
        ax.axvspan(0.0, posting_end, color="#c9a227", alpha=0.12, linewidth=0, label="posting")
    ax.axvline(float(cfg.fill_start_seconds), color="#b83232", linewidth=1.0, alpha=0.75, label="first fill")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5, zorder=0)
    ax.set_xlim(plot_start, plot_end)
    title_suffix = "" if filled_label is None else f", {filled_label} filled orders"
    ax.set_title(f"Scheduled passive impact cost paths{title_suffix}")
    ax.set_xlabel("seconds from first posted limit")
    ax.set_ylabel("cumulative impact cost")
    ax_impact.set_ylabel("price impact")
    ax_impact.tick_params(axis="y", colors="#5d6872")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    impact_handles, impact_labels = ax_impact.get_legend_handles_labels()
    ax.legend(handles + impact_handles, labels + impact_labels, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_end_seconds(cfg: ScheduledPassiveCostConfig) -> float:
    """Return a focused plot horizon around the scheduled posting/fill region."""
    n_fills = int(cfg.n_posting_slots if cfg.n_filled_orders is None else cfg.n_filled_orders)
    if n_fills <= 0:
        last_fill = float(cfg.fill_start_seconds)
    else:
        fill_times = _scheduled_fill_times(n_fills, cfg)
        last_fill = float(fill_times[-1]) if fill_times.size else float(cfg.fill_start_seconds)
    focus_end = last_fill + 0.030
    return min(float(cfg.horizon_seconds), max(focus_end, float(cfg.fill_start_seconds) + 0.050))


def _plot_start_seconds(cfg: ScheduledPassiveCostConfig) -> float:
    """Return the left edge for the zoomed execution plot."""
    return max(0.0, float(cfg.fill_start_seconds) - 0.050)


def _episode_status(candidate: Any, *, status: str, **values: Any) -> dict[str, Any]:
    base = {
        "episode_id": int(candidate.episode_id),
        "window_start": str(candidate.window_start),
        "first_source_row_pos": int(candidate.first_source_row_pos),
        "status": status,
        "n_posted_orders": 0,
        "n_filled_orders": 0,
        "n_market_events": 0,
        "selected_qty": 0,
        "filled_qty": 0,
        "initial_q": np.nan,
        "mean_final_cost": np.nan,
        "median_final_cost": np.nan,
    }
    base.update(values)
    return base


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
    market_rows = window[(types == MARKET) & (sides == market_side)]
    if market_rows.empty:
        return np.array([], dtype=np.float64)
    return event_seconds(market_rows, origin=origin).astype(np.float64)


def _queue_to_u32(values: np.ndarray) -> np.ndarray:
    return np.rint(np.maximum(values, 0.0)).astype(np.uint32)


def _sample_previous_value(
    *,
    event_times: np.ndarray,
    event_values: np.ndarray,
    output_grid: np.ndarray,
    initial_value: float = 0.0,
) -> np.ndarray:
    times = np.asarray(event_times, dtype=np.float64)
    values = np.asarray(event_values, dtype=np.float64)
    grid = np.asarray(output_grid, dtype=np.float64)
    if times.shape != values.shape:
        raise ValueError("event_times and event_values must have the same shape")
    idx = np.searchsorted(times, grid, side="right") - 1
    out = np.full(grid.shape, float(initial_value), dtype=np.float64)
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def _output_grid(cfg: ScheduledPassiveCostConfig) -> np.ndarray:
    n_steps = int(np.floor(cfg.horizon_seconds / cfg.output_step_seconds))
    grid = np.arange(n_steps + 1, dtype=np.float64) * float(cfg.output_step_seconds)
    if grid[-1] < cfg.horizon_seconds:
        grid = np.append(grid, float(cfg.horizon_seconds))
    fill_count = int(cfg.n_posting_slots if cfg.n_filled_orders is None else cfg.n_filled_orders)
    fill_times = _scheduled_fill_times(fill_count, cfg)
    fill_times = fill_times[fill_times <= float(cfg.horizon_seconds)]
    return np.unique(np.concatenate([grid, fill_times])).astype(np.float64)


def _scheduled_fill_times(n_fills: int, cfg: ScheduledPassiveCostConfig) -> np.ndarray:
    """Return synthetic fill times for the configured execution schedule."""
    n = int(n_fills)
    if n <= 0:
        return np.array([], dtype=np.float64)

    schedule = cfg.fill_schedule.lower()
    if schedule == "regular":
        delays = np.arange(n, dtype=np.float64) * float(cfg.fill_spacing_seconds)
    elif schedule == "exponential_quantile":
        if n == 1:
            delays = np.array([0.0], dtype=np.float64)
        else:
            probabilities = np.arange(n, dtype=np.float64) / float(n)
            scale = float(cfg.fill_half_life_seconds) / np.log(2.0)
            delays = -scale * np.log1p(-probabilities)
    else:
        raise ValueError("fill_schedule must be 'regular' or 'exponential_quantile'")

    return float(cfg.fill_start_seconds) + delays


def _timestamp_like(ts: pd.Series, value: object) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if getattr(ts.dt, "tz", None) is not None and out.tzinfo is None:
        out = out.tz_localize(ts.dt.tz)
    return out


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
    except (ImportError, ModuleNotFoundError):
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        code_python = repo_root / "code" / "python"
        if str(code_python) not in sys.path:
            sys.path.insert(0, str(code_python))
        import simproj  # type: ignore

        return simproj


def _validate_config(cfg: ScheduledPassiveCostConfig) -> None:
    validate_passive_impact_model_config(_passive_impact_model_config(cfg))
    if cfg.episode_spacing_seconds < 0:
        raise ValueError("episode_spacing_seconds must be nonnegative")
    if cfg.max_episodes is not None and cfg.max_episodes <= 0:
        raise ValueError("max_episodes must be positive or None")
    if cfg.posting_spacing_seconds <= 0:
        raise ValueError("posting_spacing_seconds must be positive")
    if cfg.n_posting_slots <= 0:
        raise ValueError("n_posting_slots must be positive")
    if cfg.fill_start_seconds < 0:
        raise ValueError("fill_start_seconds must be nonnegative")
    if cfg.fill_schedule not in {"regular", "exponential_quantile"}:
        raise ValueError("fill_schedule must be 'regular' or 'exponential_quantile'")
    if cfg.fill_spacing_seconds < 0:
        raise ValueError("fill_spacing_seconds must be nonnegative")
    if cfg.fill_half_life_seconds <= 0:
        raise ValueError("fill_half_life_seconds must be positive")
    if cfg.n_filled_orders is not None and cfg.n_filled_orders < 0:
        raise ValueError("n_filled_orders must be nonnegative or None")
    if cfg.horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if cfg.output_step_seconds <= 0:
        raise ValueError("output_step_seconds must be positive")
    if cfg.fill_start_seconds > cfg.horizon_seconds:
        raise ValueError("fill_start_seconds must not exceed horizon_seconds")
    if cfg.n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if cfg.min_market_events < 0:
        raise ValueError("min_market_events must be nonnegative")
    if len(cfg.propagator_weights) != len(cfg.propagator_beta):
        raise ValueError("propagator_weights and propagator_beta must have matching lengths")
    if any(float(beta) <= 0.0 for beta in cfg.propagator_beta):
        raise ValueError("propagator_beta values must be positive")


def _passive_impact_model_config(cfg: ScheduledPassiveCostConfig) -> PassiveImpactModelConfig:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated-path", default=ScheduledPassiveCostConfig.aggregated_path)
    parser.add_argument("--output-dir", default=ScheduledPassiveCostConfig.output_dir)
    parser.add_argument("--episode-spacing-seconds", type=float, default=ScheduledPassiveCostConfig.episode_spacing_seconds)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--randomize-episodes", action="store_true")
    parser.add_argument("--posting-spacing-seconds", type=float, default=ScheduledPassiveCostConfig.posting_spacing_seconds)
    parser.add_argument("--n-posting-slots", type=int, default=ScheduledPassiveCostConfig.n_posting_slots)
    parser.add_argument("--fill-start-seconds", type=float, default=ScheduledPassiveCostConfig.fill_start_seconds)
    parser.add_argument(
        "--fill-schedule",
        choices=["regular", "exponential_quantile"],
        default=ScheduledPassiveCostConfig.fill_schedule,
    )
    parser.add_argument("--fill-spacing-seconds", type=float, default=ScheduledPassiveCostConfig.fill_spacing_seconds)
    parser.add_argument("--fill-half-life-seconds", type=float, default=ScheduledPassiveCostConfig.fill_half_life_seconds)
    parser.add_argument("--n-filled-orders", type=int, default=None)
    parser.add_argument("--horizon-seconds", type=float, default=ScheduledPassiveCostConfig.horizon_seconds)
    parser.add_argument("--output-step-seconds", type=float, default=ScheduledPassiveCostConfig.output_step_seconds)
    parser.add_argument("--n-simulations", type=int, default=ScheduledPassiveCostConfig.n_simulations)
    parser.add_argument("--allow-incomplete-posting-grid", action="store_true")
    parser.add_argument("--min-market-events", type=int, default=ScheduledPassiveCostConfig.min_market_events)
    parser.add_argument("--raw-side", default=ScheduledPassiveCostConfig.raw_side)
    parser.add_argument("--queue-col", default=ScheduledPassiveCostConfig.queue_col)
    parser.add_argument("--market-side", default=ScheduledPassiveCostConfig.market_side)
    parser.add_argument("--start-time", default=ScheduledPassiveCostConfig.start_time)
    parser.add_argument("--end-time", default=ScheduledPassiveCostConfig.end_time)
    parser.add_argument("--seed", type=int, default=ScheduledPassiveCostConfig.seed)
    parser.add_argument("--a-l", type=float, default=ScheduledPassiveCostConfig.a_l)
    parser.add_argument("--b-l", type=float, default=ScheduledPassiveCostConfig.b_l)
    parser.add_argument("--a-c", type=float, default=ScheduledPassiveCostConfig.a_c)
    parser.add_argument("--b-c", type=float, default=ScheduledPassiveCostConfig.b_c)
    parser.add_argument(
        "--impact-model",
        choices=["reduced_form", "tail_propagator", "propagator_tail", "structural"],
        default=ScheduledPassiveCostConfig.impact_model,
    )
    parser.add_argument("--propagator-kappa", type=float, default=ScheduledPassiveCostConfig.propagator_kappa)
    parser.add_argument("--propagator-gamma", type=float, default=ScheduledPassiveCostConfig.propagator_gamma)
    parser.add_argument(
        "--propagator-weights",
        default=",".join(str(x) for x in ScheduledPassiveCostConfig.propagator_weights),
    )
    parser.add_argument(
        "--propagator-beta",
        default=",".join(str(x) for x in ScheduledPassiveCostConfig.propagator_beta),
    )
    parser.add_argument("--propagator-tail-zeta", type=float, default=ScheduledPassiveCostConfig.propagator_tail_zeta)
    parser.add_argument("--c-kappa", type=float, default=ScheduledPassiveCostConfig.c_kappa)
    parser.add_argument("--hawkes-mu", type=float, default=ScheduledPassiveCostConfig.hawkes_mu)
    parser.add_argument("--hawkes-alpha", default=",".join(str(x) for x in ScheduledPassiveCostConfig.hawkes_alpha))
    parser.add_argument("--hawkes-beta", default=",".join(str(x) for x in ScheduledPassiveCostConfig.hawkes_beta))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ScheduledPassiveCostConfig(
        aggregated_path=args.aggregated_path,
        output_dir=args.output_dir,
        episode_spacing_seconds=args.episode_spacing_seconds,
        max_episodes=args.max_episodes,
        randomize_episodes=args.randomize_episodes,
        posting_spacing_seconds=args.posting_spacing_seconds,
        n_posting_slots=args.n_posting_slots,
        fill_start_seconds=args.fill_start_seconds,
        fill_schedule=args.fill_schedule,
        fill_spacing_seconds=args.fill_spacing_seconds,
        fill_half_life_seconds=args.fill_half_life_seconds,
        n_filled_orders=args.n_filled_orders,
        horizon_seconds=args.horizon_seconds,
        output_step_seconds=args.output_step_seconds,
        n_simulations=args.n_simulations,
        require_full_posting_grid=not args.allow_incomplete_posting_grid,
        min_market_events=args.min_market_events,
        raw_side=args.raw_side,
        queue_col=args.queue_col,
        market_side=args.market_side,
        start_time=args.start_time,
        end_time=args.end_time,
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
    summary = run_scheduled_passive_cost_pipeline(cfg)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2))


def _parse_floats(raw: str) -> tuple[float, ...]:
    if not raw:
        return ()
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
