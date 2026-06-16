"""Looped passive post/fill/cancel impact-cost experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from experiments.plot_utils_common import (
        add_format_argument,
        add_title_argument,
        with_output_format,
    )
    from experiments.impact_cost.core.cost_utils import MARKET, event_seconds
    from experiments.impact_cost.core.experiment_utils import (
        candidate_episodes,
        load_aggregated_depth,
        queue_to_u32,
        sample_previous_value,
        timestamp_like,
        window_from_aggregated,
    )
    from experiments.impact_cost.core.level_execution import market_side_for_queue
    from experiments.impact_cost.core.passive_impact import (
        PassiveImpactModelConfig,
        execution_cost_jump_series,
        passive_impact_path_from_queue_samples,
        validate_passive_impact_model_config,
    )
    from experiments.impact_cost.core.passive_lifecycle import (
        PassiveLifecycleConfig,
        active_displacement_at_times,
        generate_passive_lifecycle,
        validate_passive_lifecycle_config,
    )
    from experiments.impact_cost.core.reduced_form_impact import (
        DEFAULT_PROPAGATOR_BETA,
        DEFAULT_PROPAGATOR_GAMMA,
        DEFAULT_PROPAGATOR_KAPPA,
        DEFAULT_PROPAGATOR_WEIGHTS,
    )
else:
    from ...plot_utils_common import (
        add_format_argument,
        add_title_argument,
        with_output_format,
    )
    from ..core.cost_utils import MARKET, event_seconds
    from ..core.experiment_utils import (
        candidate_episodes,
        load_aggregated_depth,
        queue_to_u32,
        sample_previous_value,
        timestamp_like,
        window_from_aggregated,
    )
    from ..core.level_execution import market_side_for_queue
    from ..core.passive_impact import (
        PassiveImpactModelConfig,
        execution_cost_jump_series,
        passive_impact_path_from_queue_samples,
        validate_passive_impact_model_config,
    )
    from ..core.passive_lifecycle import (
        PassiveLifecycleConfig,
        active_displacement_at_times,
        generate_passive_lifecycle,
        validate_passive_lifecycle_config,
    )
    from ..core.reduced_form_impact import (
        DEFAULT_PROPAGATOR_BETA,
        DEFAULT_PROPAGATOR_GAMMA,
        DEFAULT_PROPAGATOR_KAPPA,
        DEFAULT_PROPAGATOR_WEIGHTS,
    )


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class LifecyclePassiveCostConfig:
    """Configuration for looped passive lifecycle cost paths."""

    aggregated_path: str = "experiments/impact_cost/load_experiments/data/processed/factual_2025_05_29_esm5.parquet"
    output_dir: str = "experiments/impact_cost/load_experiments/data/lifecycle_passive_cost"
    image_dir: str = "experiments/impact_cost/load_experiments/images"
    episode_spacing_seconds: float = 60.0
    max_episodes: int | None = None
    randomize_episodes: bool = False
    horizon_seconds: float = 3.0
    output_step_seconds: float = 0.002
    warmup_seconds: float = 0.0
    n_policy_paths: int = 50
    min_market_events: int = 1
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    start_time: str | None = None
    end_time: str | None = None
    seed: int = 2027
    n_cycles: int = 3
    orders_per_cycle: int = 10
    order_qty: int = 1
    posting_spacing_seconds: float = 0.010
    fill_count_model: str = "binomial"
    fill_probability: float = 1.0 / 7.0
    fixed_filled_orders: int | None = None
    fill_selection: str = "oldest"
    fill_time_model: str = "clustered_exponential"
    fill_wait_mean_seconds: float = 0.150
    fill_gap_mean_seconds: float = 0.010
    min_resting_seconds: float = 0.300
    cancel_delay_seconds: float = 0.010
    cancel_jitter_seconds: float = 0.001
    repost_delay_seconds: float = 0.050
    propagator_kappa: float = DEFAULT_PROPAGATOR_KAPPA
    propagator_gamma: float = DEFAULT_PROPAGATOR_GAMMA
    propagator_weights: tuple[float, ...] = DEFAULT_PROPAGATOR_WEIGHTS
    propagator_beta: tuple[float, ...] = DEFAULT_PROPAGATOR_BETA
    propagator_tail_zeta: float = 0.0
    b_l: float = -0.000097
    b_c: float = 0.0000989


def run_lifecycle_passive_cost_pipeline(
    cfg: LifecyclePassiveCostConfig,
    *,
    include_title: bool = False,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """Run looped passive lifecycle cost paths and write outputs."""
    _validate_config(cfg)
    output_dir = Path(cfg.output_dir)
    image_dir = Path(cfg.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    aggregated = load_aggregated_depth(cfg.aggregated_path)
    candidates = candidate_episodes(aggregated, cfg)
    output_grid = _output_grid(cfg)
    impact_cfg = _passive_impact_model_config(cfg)
    lifecycle_cfg = _lifecycle_config(cfg)
    simproj = _import_simproj()

    episode_rows: list[dict[str, Any]] = []
    path_summary_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    cancel_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    jump_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    impact_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []

    for candidate in candidates.itertuples(index=False):
        episode_id = int(candidate.episode_id)
        window_start = pd.Timestamp(candidate.window_start)
        window = window_from_aggregated(
            aggregated,
            window_start,
            horizon_seconds=cfg.horizon_seconds,
            warmup_seconds=cfg.warmup_seconds,
        )
        if window.empty:
            episode_rows.append(_episode_status(candidate, status="empty_window"))
            continue

        consuming_side = market_side_for_queue(
            raw_side=cfg.raw_side,
            queue_col=cfg.queue_col,
            market_side=cfg.market_side,
        )
        market = _market_samples(
            window,
            origin=window_start,
            market_side=consuming_side,
            queue_col=cfg.queue_col,
        )
        n_warmup_market_events = int((market["time_s"] < 0.0).sum()) if len(market) else 0
        n_posting_market_events = int((market["time_s"] >= 0.0).sum()) if len(market) else 0
        if n_posting_market_events < int(cfg.min_market_events):
            episode_rows.append(
                _episode_status(
                    candidate,
                    status="too_few_market_events",
                    n_market_events=n_posting_market_events,
                    n_warmup_market_events=n_warmup_market_events,
                )
            )
            continue

        market_times = market["time_s"].to_numpy(dtype=np.float64)
        impact_market_times = _impact_clock_times(
            market_times,
            warmup_seconds=float(cfg.warmup_seconds),
        )
        q_bar = queue_to_u32(market["q_bar"].to_numpy(dtype=np.float64))
        final_costs: list[float] = []
        final_impacts: list[float] = []
        total_filled: list[int] = []
        total_canceled: list[int] = []
        total_clipped: list[int] = []

        for policy_path_id in range(int(cfg.n_policy_paths)):
            lifecycle = generate_passive_lifecycle(
                lifecycle_cfg,
                seed=int(cfg.seed) + episode_id * 100_003 + policy_path_id,
                episode_id=episode_id,
                policy_path_id=policy_path_id,
            )
            orders = _tag_path_frame(lifecycle["orders"], window_start)
            fills_all = _tag_path_frame(lifecycle["fills"], window_start)
            cancels_all = _tag_path_frame(lifecycle["cancels"], window_start)
            events_all = _tag_path_frame(lifecycle["events"], window_start)
            cycles = _tag_path_frame(lifecycle["cycle_summary"], window_start)
            fills = fills_all[fills_all["fill_time_s"] <= cfg.horizon_seconds].copy()
            cancels = cancels_all[
                cancels_all["cancel_time_s"] <= cfg.horizon_seconds
            ].copy()
            events = events_all[events_all["time_s"] <= cfg.horizon_seconds].copy()

            active_market_qty = active_displacement_at_times(events, market_times)
            q_no_us = queue_to_u32(q_bar.astype(np.float64) - active_market_qty)
            clipped = int(np.sum(active_market_qty > q_bar.astype(np.float64)))
            impact_at_markets = passive_impact_path_from_queue_samples(
                q_no_us,
                q_bar,
                impact_market_times,
                queue_col=cfg.queue_col,
                cfg=impact_cfg,
                simproj=simproj,
            )
            impact_on_grid = sample_previous_value(
                event_times=market_times,
                event_values=impact_at_markets,
                output_grid=output_grid,
                initial_value=0.0,
            )
            active_on_grid = active_displacement_at_times(events, output_grid)

            jumps = execution_cost_jump_series(
                fills,
                market_times=market_times,
                impact=impact_at_markets,
                time_col="fill_time_s",
                qty_col="qty",
                copy_columns=("cycle_id", "order_id", "order_slot", "post_time_s"),
                extra_columns={
                    "episode_id": episode_id,
                    "policy_path_id": policy_path_id,
                    "window_start": str(window_start),
                },
            )
            cumulative_path = sample_previous_value(
                event_times=jumps["fill_time_s"].to_numpy(dtype=np.float64),
                event_values=jumps["cumulative_cost"].to_numpy(dtype=np.float64),
                output_grid=output_grid,
                initial_value=0.0,
            )
            n_filled = int(len(fills))
            n_canceled = int(len(cancels))
            n_generated_filled = int(len(fills_all))
            n_generated_canceled = int(len(cancels_all))
            n_open_at_horizon = int(len(orders) - n_filled - n_canceled)
            final_cost = float(cumulative_path[-1]) if len(cumulative_path) else 0.0
            final_impact = float(impact_on_grid[-1]) if len(impact_on_grid) else 0.0
            max_active_qty = float(np.max(active_on_grid)) if len(active_on_grid) else 0.0
            final_costs.append(final_cost)
            final_impacts.append(final_impact)
            total_filled.append(n_filled)
            total_canceled.append(n_canceled)
            total_clipped.append(clipped)

            order_rows.extend(orders.to_dict("records"))
            fill_rows.extend(fills_all.to_dict("records"))
            cancel_rows.extend(cancels_all.to_dict("records"))
            event_rows.extend(events_all.to_dict("records"))
            cycle_rows.extend(cycles.to_dict("records"))
            jump_rows.extend(jumps.to_dict("records"))
            path_summary_rows.append(
                {
                    "episode_id": episode_id,
                    "policy_path_id": policy_path_id,
                    "window_start": str(window_start),
                    "n_posted_orders": int(len(orders)),
                    "n_filled_orders": n_filled,
                    "n_canceled_orders": n_canceled,
                    "n_generated_filled_orders": n_generated_filled,
                    "n_generated_canceled_orders": n_generated_canceled,
                    "n_open_orders_at_horizon": n_open_at_horizon,
                    "n_clipped_market_samples": clipped,
                    "max_active_qty": max_active_qty,
                    "final_cost": final_cost,
                    "final_price_impact": final_impact,
                }
            )
            path_rows.extend(
                {
                    "episode_id": episode_id,
                    "policy_path_id": policy_path_id,
                    "window_start": str(window_start),
                    "time_s": float(t),
                    "cumulative_cost": float(value),
                    "n_filled_orders": n_filled,
                    "n_canceled_orders": n_canceled,
                }
                for t, value in zip(output_grid, cumulative_path)
            )
            impact_rows.extend(
                {
                    "episode_id": episode_id,
                    "policy_path_id": policy_path_id,
                    "window_start": str(window_start),
                    "time_s": float(t),
                    "price_impact": float(value),
                    "n_filled_orders": n_filled,
                    "n_canceled_orders": n_canceled,
                }
                for t, value in zip(output_grid, impact_on_grid)
            )
            active_rows.extend(
                {
                    "episode_id": episode_id,
                    "policy_path_id": policy_path_id,
                    "window_start": str(window_start),
                    "time_s": float(t),
                    "active_qty": float(value),
                    "n_filled_orders": n_filled,
                    "n_canceled_orders": n_canceled,
                }
                for t, value in zip(output_grid, active_on_grid)
            )

        episode_rows.append(
            _episode_status(
                candidate,
                status="ok",
                n_market_events=n_posting_market_events,
                n_warmup_market_events=n_warmup_market_events,
                n_policy_paths=int(cfg.n_policy_paths),
                mean_final_cost=float(np.mean(final_costs)) if final_costs else 0.0,
                median_final_cost=float(np.median(final_costs)) if final_costs else 0.0,
                mean_final_price_impact=float(np.mean(final_impacts)) if final_impacts else 0.0,
                mean_filled_orders=float(np.mean(total_filled)) if total_filled else 0.0,
                mean_canceled_orders=float(np.mean(total_canceled)) if total_canceled else 0.0,
                mean_clipped_market_samples=float(np.mean(total_clipped)) if total_clipped else 0.0,
            )
        )

    episodes = pd.DataFrame(episode_rows)
    path_summary = pd.DataFrame(path_summary_rows)
    orders = pd.DataFrame(order_rows)
    fills = pd.DataFrame(fill_rows)
    cancels = pd.DataFrame(cancel_rows)
    events = pd.DataFrame(event_rows)
    cycles = pd.DataFrame(cycle_rows)
    jumps = pd.DataFrame(jump_rows)
    samples = pd.DataFrame(path_rows)
    impact_samples = pd.DataFrame(impact_rows)
    active_samples = pd.DataFrame(active_rows)
    cost_summary = _summarize_cost_paths(samples)
    cost_summary_by_fill_count = _summarize_cost_paths(samples, by_fill_count=True)
    impact_summary = _summarize_impact_paths(impact_samples)
    active_summary = _summarize_active_paths(active_samples)

    episodes.to_csv(output_dir / "episode_summary.csv", index=False)
    path_summary.to_csv(output_dir / "policy_path_summary.csv", index=False)
    orders.to_csv(output_dir / "policy_orders.csv", index=False)
    fills.to_csv(output_dir / "policy_fills.csv", index=False)
    cancels.to_csv(output_dir / "policy_cancels.csv", index=False)
    events.to_csv(output_dir / "policy_events.csv", index=False)
    cycles.to_csv(output_dir / "policy_cycle_summary.csv", index=False)
    jumps.to_csv(output_dir / "impact_cost_fill_jumps.csv", index=False)
    samples.to_csv(output_dir / "impact_cost_path_samples.csv", index=False)
    cost_summary.to_csv(output_dir / "impact_cost_path_summary.csv", index=False)
    cost_summary_by_fill_count.to_csv(
        output_dir / "impact_cost_path_summary_by_fill_count.csv",
        index=False,
    )
    impact_samples.to_csv(output_dir / "price_impact_path_samples.csv", index=False)
    impact_summary.to_csv(output_dir / "price_impact_path_summary.csv", index=False)
    active_samples.to_csv(output_dir / "active_quantity_path_samples.csv", index=False)
    active_summary.to_csv(output_dir / "active_quantity_path_summary.csv", index=False)
    plot_path = with_output_format(
        image_dir / "lifecycle_impact_cost_paths.pdf",
        output_format,
    )
    step_plot_path = with_output_format(
        image_dir / "lifecycle_representative_cost_steps.pdf",
        output_format,
    )
    step_shared_y_plot_path = with_output_format(
        image_dir / "lifecycle_representative_cost_steps_shared_y.pdf",
        output_format,
    )

    _plot_lifecycle_paths(
        plot_path,
        samples,
        cost_summary,
        impact_summary,
        active_summary,
        cfg,
        include_title=include_title,
    )
    _plot_representative_step_paths(
        step_plot_path,
        samples,
        jumps,
        path_summary,
        cfg,
        include_title=include_title,
    )
    _plot_representative_step_paths(
        step_shared_y_plot_path,
        samples,
        jumps,
        path_summary,
        cfg,
        shared_y=True,
        include_title=include_title,
    )

    with open(output_dir / "lifecycle_passive_cost_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    ok = episodes[episodes["status"] == "ok"] if not episodes.empty else episodes
    return {
        "output_dir": output_dir,
        "n_candidate_episodes": int(len(candidates)),
        "n_ok_episodes": int(len(ok)),
        "n_policy_paths": int(len(path_summary)),
        "n_fill_jumps": int(len(jumps)),
        "mean_final_cost": float(path_summary["final_cost"].mean())
        if len(path_summary)
        else float("nan"),
        "median_final_cost": float(path_summary["final_cost"].median())
        if len(path_summary)
        else float("nan"),
        "mean_filled_orders": float(path_summary["n_filled_orders"].mean())
        if len(path_summary)
        else float("nan"),
        "plot_path": plot_path,
        "step_plot_path": step_plot_path,
        "step_shared_y_plot_path": step_shared_y_plot_path,
        "path_summary_path": output_dir / "policy_path_summary.csv",
        "fill_jumps_path": output_dir / "impact_cost_fill_jumps.csv",
    }


def _market_samples(
    window: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    market_side: str,
    queue_col: str,
) -> pd.DataFrame:
    types = window["order_type"].astype(str).str.lower()
    sides = window["side"].astype(str)
    rows = window[(types == MARKET) & (sides == market_side)].copy()
    if rows.empty:
        return pd.DataFrame(
            {
                "time_s": pd.Series(dtype=np.float64),
                "q_bar": pd.Series(dtype=np.float64),
            }
        )
    return pd.DataFrame(
        {
            "time_s": event_seconds(rows, origin=origin).astype(np.float64),
            "q_bar": rows[queue_col].to_numpy(dtype=np.float64),
        }
    )


def _impact_clock_times(
    market_times: np.ndarray,
    *,
    warmup_seconds: float,
) -> np.ndarray:
    """Return nonnegative market times for native tail-state updates."""
    times = np.asarray(market_times, dtype=np.float64)
    shifted = times + float(warmup_seconds)
    if shifted.size:
        shifted = np.maximum(shifted, 0.0)
    return shifted


def _tag_path_frame(frame: pd.DataFrame, window_start: pd.Timestamp) -> pd.DataFrame:
    out = frame.copy()
    if "window_start" not in out.columns:
        out.insert(2, "window_start", str(window_start))
    return out


def _output_grid(cfg: LifecyclePassiveCostConfig) -> np.ndarray:
    n_steps = int(np.floor(float(cfg.horizon_seconds) / float(cfg.output_step_seconds)))
    grid = np.arange(n_steps + 1, dtype=np.float64) * float(cfg.output_step_seconds)
    if grid[-1] < float(cfg.horizon_seconds):
        grid = np.append(grid, float(cfg.horizon_seconds))
    return np.unique(grid).astype(np.float64)


def _summarize_cost_paths(
    samples: pd.DataFrame,
    *,
    by_fill_count: bool = False,
) -> pd.DataFrame:
    group_cols = ["time_s"] if not by_fill_count else ["n_filled_orders", "time_s"]
    columns = [
        *([] if not by_fill_count else ["n_filled_orders"]),
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
    grouped = samples.groupby(group_cols, sort=True)["cumulative_cost"]
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


def _summarize_impact_paths(samples: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "time_s",
        "n_samples",
        "mean_price_impact",
        "std_price_impact",
        "q05_price_impact",
        "q95_price_impact",
    ]
    if samples.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})
    grouped = samples.groupby("time_s", sort=True)["price_impact"]
    summary = grouped.agg(
        n_samples="count",
        mean_price_impact="mean",
        std_price_impact="std",
        q05_price_impact=lambda x: float(np.quantile(x, 0.05)),
        q95_price_impact=lambda x: float(np.quantile(x, 0.95)),
    ).reset_index()
    return summary[columns]


def _summarize_active_paths(samples: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "time_s",
        "n_samples",
        "mean_active_qty",
        "q05_active_qty",
        "q95_active_qty",
    ]
    if samples.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})
    grouped = samples.groupby("time_s", sort=True)["active_qty"]
    summary = grouped.agg(
        n_samples="count",
        mean_active_qty="mean",
        q05_active_qty=lambda x: float(np.quantile(x, 0.05)),
        q95_active_qty=lambda x: float(np.quantile(x, 0.95)),
    ).reset_index()
    return summary[columns]


def _plot_lifecycle_paths(
    path: Path,
    _samples: pd.DataFrame,
    cost_summary: pd.DataFrame,
    impact_summary: pd.DataFrame,
    _active_summary: pd.DataFrame,
    cfg: LifecyclePassiveCostConfig,
    *,
    include_title: bool = False,
) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax_impact = ax.twinx()
    ax_impact.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)

    if not impact_summary.empty:
        x = impact_summary["time_s"].to_numpy(dtype=np.float64)
        mean = impact_summary["mean_price_impact"].to_numpy(dtype=np.float64)
        ax_impact.plot(
            x,
            mean,
            color="#5d6872",
            linewidth=1.9,
            linestyle="--",
            alpha=0.78,
            label="Mean price impact",
        )

    if not cost_summary.empty:
        x = cost_summary["time_s"].to_numpy(dtype=np.float64)
        mean = cost_summary["mean_cost"].to_numpy(dtype=np.float64)
        ax.step(x, mean, where="post", color="#1f5d99", linewidth=2.0, label="Mean cost")

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlim(0.0, float(cfg.horizon_seconds))
    if include_title:
        ax.set_title("Looped passive lifecycle mean impact cost")
    ax.set_xlabel("Seconds from first post")
    ax.set_ylabel("Cumulative impact cost")
    ax_impact.set_ylabel("Price impact")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax_impact.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_representative_step_paths(
    path: Path,
    samples: pd.DataFrame,
    jumps: pd.DataFrame,
    path_summary: pd.DataFrame,
    cfg: LifecyclePassiveCostConfig,
    *,
    n_windows: int = 3,
    shared_y: bool = False,
    include_title: bool = False,
) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    selected = _select_representative_step_paths(path_summary, n_windows=n_windows)
    if selected.empty:
        fig, ax = plt.subplots(figsize=(10.0, 3.2))
        ax.text(0.5, 0.5, "no individual cost paths", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
        return

    n_panels = int(len(selected))
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10.8, max(3.0, 2.55 * n_panels)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    y_max = _representative_y_max(samples, selected) if shared_y else np.nan

    for ax, row in zip(axes, selected.itertuples(index=False)):
        episode_id = int(row.episode_id)
        policy_path_id = int(row.policy_path_id)
        path_rows = samples[
            (samples["episode_id"] == episode_id)
            & (samples["policy_path_id"] == policy_path_id)
        ].sort_values("time_s")
        jump_rows = jumps[
            (jumps["episode_id"] == episode_id)
            & (jumps["policy_path_id"] == policy_path_id)
        ].sort_values("fill_time_s")

        if not path_rows.empty:
            ax.step(
                path_rows["time_s"],
                path_rows["cumulative_cost"],
                where="post",
                color="#1f5d99",
                linewidth=2.0,
                label="Cost path",
            )
        if not jump_rows.empty:
            ax.scatter(
                jump_rows["fill_time_s"],
                jump_rows["cumulative_cost"],
                s=24,
                color="#b83232",
                zorder=3,
                label="Fills",
            )
            for fill_time in jump_rows["fill_time_s"].to_numpy(dtype=np.float64):
                ax.axvline(fill_time, color="#b83232", linewidth=0.6, alpha=0.16)

        ax.axhline(0.0, color="black", linewidth=0.75, alpha=0.45)
        if np.isfinite(y_max):
            ax.set_ylim(min(0.0, -0.04 * y_max), 1.08 * y_max)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Cost")
        if include_title:
            ax.set_title(
                "window {episode}, policy path {policy}, fills {fills}, final {final:.6g}".format(
                    episode=episode_id,
                    policy=policy_path_id,
                    fills=int(row.n_filled_orders),
                    final=float(row.final_cost),
                ),
                fontsize=10,
            )
        ax.legend(loc="best")

    axes[-1].set_xlabel("Seconds from first post")
    axes[-1].set_xlim(0.0, float(cfg.horizon_seconds))
    title = "Representative individual lifecycle cost paths"
    if shared_y:
        title += " (shared y-axis)"
    if include_title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _select_representative_step_paths(
    path_summary: pd.DataFrame,
    *,
    n_windows: int,
) -> pd.DataFrame:
    columns = [
        "episode_id",
        "policy_path_id",
        "n_filled_orders",
        "final_cost",
    ]
    if path_summary.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})

    candidates = path_summary[path_summary["n_filled_orders"] > 0].copy()
    if candidates.empty:
        candidates = path_summary.copy()

    rows: list[pd.Series] = []
    for episode_id in sorted(candidates["episode_id"].unique())[: int(n_windows)]:
        frame = candidates[candidates["episode_id"] == episode_id].copy()
        target = float(frame["final_cost"].median())
        idx = (frame["final_cost"] - target).abs().idxmin()
        rows.append(frame.loc[idx])

    if not rows:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})
    out = pd.DataFrame(rows)
    return out[columns]


def _representative_y_max(samples: pd.DataFrame, selected: pd.DataFrame) -> float:
    values: list[float] = []
    for row in selected.itertuples(index=False):
        path_rows = samples[
            (samples["episode_id"] == int(row.episode_id))
            & (samples["policy_path_id"] == int(row.policy_path_id))
        ]
        if not path_rows.empty:
            values.append(float(path_rows["cumulative_cost"].max()))
    if not values:
        return np.nan
    return max(max(values), 1e-12)


def _episode_status(candidate: Any, *, status: str, **values: Any) -> dict[str, Any]:
    base = {
        "episode_id": int(candidate.episode_id),
        "window_start": str(candidate.window_start),
        "first_source_row_pos": int(candidate.first_source_row_pos),
        "status": status,
        "n_market_events": 0,
        "n_warmup_market_events": 0,
        "n_policy_paths": 0,
        "mean_final_cost": np.nan,
        "median_final_cost": np.nan,
        "mean_final_price_impact": np.nan,
        "mean_filled_orders": np.nan,
        "mean_canceled_orders": np.nan,
        "mean_clipped_market_samples": np.nan,
    }
    base.update(values)
    return base


def _setup_matplotlib() -> None:
    mpl_cache = Path("/private/tmp/matplotlib-cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")


def _passive_impact_model_config(cfg: LifecyclePassiveCostConfig) -> PassiveImpactModelConfig:
    return PassiveImpactModelConfig(
        impact_model="tail_propagator",
        propagator_kappa=cfg.propagator_kappa,
        propagator_gamma=cfg.propagator_gamma,
        propagator_weights=cfg.propagator_weights,
        propagator_beta=cfg.propagator_beta,
        propagator_tail_zeta=cfg.propagator_tail_zeta,
        b_l=cfg.b_l,
        b_c=cfg.b_c,
    )


def _lifecycle_config(cfg: LifecyclePassiveCostConfig) -> PassiveLifecycleConfig:
    return PassiveLifecycleConfig(
        n_cycles=cfg.n_cycles,
        orders_per_cycle=cfg.orders_per_cycle,
        order_qty=cfg.order_qty,
        posting_spacing_seconds=cfg.posting_spacing_seconds,
        fill_count_model=cfg.fill_count_model,  # type: ignore[arg-type]
        fill_probability=cfg.fill_probability,
        fixed_filled_orders=cfg.fixed_filled_orders,
        fill_selection=cfg.fill_selection,  # type: ignore[arg-type]
        fill_time_model=cfg.fill_time_model,  # type: ignore[arg-type]
        fill_wait_mean_seconds=cfg.fill_wait_mean_seconds,
        fill_gap_mean_seconds=cfg.fill_gap_mean_seconds,
        min_resting_seconds=cfg.min_resting_seconds,
        cancel_delay_seconds=cfg.cancel_delay_seconds,
        cancel_jitter_seconds=cfg.cancel_jitter_seconds,
        repost_delay_seconds=cfg.repost_delay_seconds,
    )


def _validate_config(cfg: LifecyclePassiveCostConfig) -> None:
    validate_passive_impact_model_config(_passive_impact_model_config(cfg))
    validate_passive_lifecycle_config(_lifecycle_config(cfg))
    if cfg.episode_spacing_seconds < 0.0:
        raise ValueError("episode_spacing_seconds must be nonnegative")
    if cfg.max_episodes is not None and cfg.max_episodes <= 0:
        raise ValueError("max_episodes must be positive or None")
    if cfg.horizon_seconds <= 0.0:
        raise ValueError("horizon_seconds must be positive")
    if cfg.output_step_seconds <= 0.0:
        raise ValueError("output_step_seconds must be positive")
    if cfg.warmup_seconds < 0.0:
        raise ValueError("warmup_seconds must be nonnegative")
    if cfg.n_policy_paths <= 0:
        raise ValueError("n_policy_paths must be positive")
    if cfg.min_market_events < 0:
        raise ValueError("min_market_events must be nonnegative")
    if cfg.start_time is not None:
        timestamp_like(pd.Series(pd.to_datetime(["2000-01-01"])), cfg.start_time)
    if cfg.end_time is not None:
        timestamp_like(pd.Series(pd.to_datetime(["2000-01-01"])), cfg.end_time)


def load_lifecycle_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    overrides: dict[str, Any] | None = None,
) -> LifecyclePassiveCostConfig:
    """Load the canonical lifecycle config from TOML or JSON."""
    path = Path(config_path)
    raw = _read_config(path)
    values = _flatten_config(raw)
    valid_fields = {field.name for field in fields(LifecyclePassiveCostConfig)}
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        raise ValueError(f"unknown lifecycle config keys: {', '.join(unknown)}")

    tuple_fields = {"propagator_weights", "propagator_beta"}
    for key in tuple_fields:
        if key in values and values[key] is not None:
            values[key] = tuple(values[key])

    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value
    return LifecyclePassiveCostConfig(**values)


def _read_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".toml":
        import tomllib

        return tomllib.loads(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            values.update(value)
        else:
            values[key] = value
    return values


def _import_simproj():
    try:
        import simproj  # type: ignore

        return simproj
    except (ImportError, ModuleNotFoundError):
        import sys

        for name in [
            key
            for key in sys.modules
            if key == "simproj" or key.startswith("simproj.")
        ]:
            sys.modules.pop(name, None)
        repo_root = Path(__file__).resolve().parents[3]
        code_python = repo_root / "code" / "python"
        sys.path = [path for path in sys.path if path != str(code_python)]
        sys.path.insert(0, str(code_python))
        import simproj  # type: ignore

        return simproj


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--n-policy-paths", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--randomize-episodes",
        action="store_true",
        default=None,
        help="Override the config and randomly sample candidate episodes.",
    )
    add_title_argument(parser, default=False)
    add_format_argument(parser, default="pdf")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_lifecycle_config(
        args.config,
        overrides={
            "output_dir": args.output_dir,
            "image_dir": args.image_dir,
            "max_episodes": args.max_episodes,
            "n_policy_paths": args.n_policy_paths,
            "seed": args.seed,
            "randomize_episodes": args.randomize_episodes,
        },
    )
    summary = run_lifecycle_passive_cost_pipeline(
        cfg,
        include_title=args.include_title,
        output_format=args.output_format,
    )
    print(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in summary.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
