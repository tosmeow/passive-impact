"""Random passive-limit sequence impact curves on factual orderbook data."""
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
from ...core.level_execution import market_side_for_queue
from ...core.passive_impact import (
    PassiveImpactModelConfig,
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
class ImpactSeriesPipelineConfig:
    """Configuration for random passive-limit impact-curve sampling.

    This pipeline is for order-of-magnitude diagnostics, not fill-conditioned
    cost attribution. It randomly selects factual limit-addition sequences,
    treats those selected rows as our passive orders, simulates no-us baseline
    queues under the anchored factual path, and writes aligned impact series.
    """

    aggregated_path: str = "experiments/impact_cost/load_experiments/data/processed/factual_2025_05_29_esm5.parquet"
    output_dir: str = "experiments/impact_cost/archive/diagnostics/data/impact_series_random_limits"
    n_orders_per_episode: int = 7
    n_episodes: int = 100
    post_span_seconds: float | None = 1.0
    horizon_seconds: float = 5.0
    output_step_seconds: float = 0.05
    n_simulations: int = 25
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    start_time: str | None = None
    end_time: str | None = None
    min_market_events: int = 1
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


def run_impact_series_pipeline(cfg: ImpactSeriesPipelineConfig) -> dict[str, Any]:
    """Sample random passive-limit sequences and aggregate impact curves."""
    _validate_config(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = _load_aggregated_depth(cfg.aggregated_path)
    episodes, selected_orders = _select_limit_sequences(aggregated, cfg)
    episodes.to_csv(output_dir / "selected_episodes.csv", index=False)
    selected_orders.to_csv(output_dir / "selected_episode_orders.csv", index=False)

    simproj = _import_simproj()
    output_grid = _output_grid(cfg)
    sample_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    for episode in episodes.itertuples(index=False):
        episode_id = int(episode.episode_id)
        window_start = pd.Timestamp(episode.window_start)
        episode_orders = selected_orders[selected_orders["episode_id"] == episode_id]
        window_samples, episode_summary = _run_one_episode(
            cfg,
            simproj,
            aggregated,
            episode_id=episode_id,
            window_start=window_start,
            selected_source_rows=set(episode_orders["source_row_pos"].astype(int)),
            output_grid=output_grid,
        )
        sample_rows.extend(window_samples)
        episode_summaries.append(
            {
                **episode._asdict(),
                **episode_summary,
            }
        )

    samples = pd.DataFrame(sample_rows)
    summary = _summarize_impact_series(samples, output_grid)
    episode_summary_frame = pd.DataFrame(episode_summaries)

    samples.to_csv(output_dir / "impact_series_samples.csv", index=False)
    summary.to_csv(output_dir / "impact_series_summary.csv", index=False)
    episode_summary_frame.to_csv(output_dir / "episode_summary.csv", index=False)
    _plot_impact_series(output_dir / "impact_series_summary.png", samples, summary, cfg)

    with open(output_dir / "impact_series_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return {
        "output_dir": output_dir,
        "n_candidate_episodes": int(episodes.attrs.get("n_candidate_episodes", len(episodes))),
        "n_selected_episodes": int(len(episodes)),
        "n_series_samples": int(len(samples)),
        "samples_path": output_dir / "impact_series_samples.csv",
        "summary_path": output_dir / "impact_series_summary.csv",
        "episodes_path": output_dir / "selected_episodes.csv",
        "episode_orders_path": output_dir / "selected_episode_orders.csv",
        "episode_summary_path": output_dir / "episode_summary.csv",
        "plot_path": output_dir / "impact_series_summary.png",
    }


def _run_one_episode(
    cfg: ImpactSeriesPipelineConfig,
    simproj: Any,
    aggregated: pd.DataFrame,
    *,
    episode_id: int,
    window_start: pd.Timestamp,
    selected_source_rows: set[int],
    output_grid: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window = _window_from_aggregated(aggregated, window_start, cfg.horizon_seconds)
    if window.empty:
        raise ValueError(f"no aggregated rows for episode window {window_start}")

    consuming_side = market_side_for_queue(
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
    )
    market_times = _market_times(window, origin=window_start, market_side=consuming_side)
    passive_flags = window["source_row_pos"].astype(int).isin(selected_source_rows).to_numpy()
    selected_qty = int(window.loc[passive_flags, "qty"].sum())
    base_summary = {
        "n_market_events": int(market_times.size),
        "selected_qty": selected_qty,
        "n_window_rows": int(len(window)),
    }

    if market_times.size < cfg.min_market_events:
        return _zero_sample_rows(episode_id, cfg.n_simulations, output_grid), {
            **base_summary,
            "status": "too_few_market_events",
        }

    initial_q = infer_initial_queue(
        window,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
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
    impact_cfg = _passive_impact_model_config(cfg)

    rows: list[dict[str, Any]] = []
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
        rows.extend(
            {
                "episode_id": episode_id,
                "simulation": sim_idx,
                "time_s": float(t),
                "impact": float(value),
            }
            for t, value in zip(output_grid, impact_on_grid)
        )

    return rows, {
        **base_summary,
        "status": "ok",
        "initial_q": int(initial_q),
        "mean_final_impact": float(np.mean([row["impact"] for row in rows if row["time_s"] == output_grid[-1]])),
    }


def _select_limit_sequences(
    aggregated: pd.DataFrame,
    cfg: ImpactSeriesPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly select factual limit-row sequences used as synthetic orders."""
    types = aggregated["order_type"].astype(str).str.lower()
    sides = aggregated["side"].astype(str)
    limit_rows = aggregated[(types == "limit") & (sides == cfg.raw_side)][
        ["source_row_pos", "ts", "qty"]
    ].copy()
    limit_rows = limit_rows.reset_index(drop=True)
    if limit_rows.empty:
        return _empty_episodes(), _empty_episode_orders()

    n = int(cfg.n_orders_per_episode)
    if len(limit_rows) < n:
        return _empty_episodes(), _empty_episode_orders()

    origin = aggregated["ts"].iloc[0]
    limit_seconds = event_seconds(limit_rows, origin=origin)
    m = len(limit_rows) - n + 1
    start_seconds = limit_seconds[:m]
    end_sequence_seconds = limit_seconds[n - 1:]
    max_data_seconds = float((aggregated["ts"].iloc[-1] - origin).total_seconds())

    candidates = start_seconds + float(cfg.horizon_seconds) <= max_data_seconds
    if cfg.post_span_seconds is not None:
        candidates &= (end_sequence_seconds - start_seconds) <= float(cfg.post_span_seconds)

    if cfg.start_time is not None:
        start_bound = _timestamp_like(limit_rows["ts"], cfg.start_time)
        candidates &= (limit_rows["ts"].iloc[:m] >= start_bound).to_numpy()
    if cfg.end_time is not None:
        end_bound = _timestamp_like(limit_rows["ts"], cfg.end_time)
        candidates &= (limit_rows["ts"].iloc[n - 1:] <= end_bound).to_numpy()

    candidate_offsets = np.flatnonzero(candidates)
    if candidate_offsets.size == 0:
        return _empty_episodes(), _empty_episode_orders()

    rng = np.random.default_rng(cfg.seed)
    n_selected = min(int(cfg.n_episodes), int(candidate_offsets.size))
    chosen_offsets = rng.choice(candidate_offsets, size=n_selected, replace=False)

    episode_rows = []
    order_rows = []
    for episode_id, offset in enumerate(chosen_offsets):
        seq = limit_rows.iloc[offset:offset + n].copy()
        window_start = pd.Timestamp(seq["ts"].iloc[0])
        seq_seconds = event_seconds(seq, origin=window_start)
        source_rows = seq["source_row_pos"].astype(int).tolist()
        episode_rows.append(
            {
                "episode_id": episode_id,
                "window_start": str(window_start),
                "first_source_row_pos": int(source_rows[0]),
                "selected_source_rows": ",".join(str(row) for row in source_rows),
                "n_selected_orders": n,
                "post_span_s": float(seq_seconds[-1] - seq_seconds[0]) if len(seq_seconds) else 0.0,
                "selected_qty": int(seq["qty"].sum()),
            }
        )
        for slot, row in enumerate(seq.itertuples(index=False)):
            order_rows.append(
                {
                    "episode_id": episode_id,
                    "order_slot": slot,
                    "source_row_pos": int(row.source_row_pos),
                    "post_ts": str(row.ts),
                    "post_time_s": float(seq_seconds[slot]),
                    "qty": int(row.qty),
                }
            )

    episodes = pd.DataFrame(episode_rows)
    episodes.attrs["n_candidate_episodes"] = int(candidate_offsets.size)
    return episodes, pd.DataFrame(order_rows)


def _summarize_impact_series(samples: pd.DataFrame, output_grid: np.ndarray) -> pd.DataFrame:
    columns = [
        "time_s",
        "n_samples",
        "mean_impact",
        "std_impact",
        "q05_impact",
        "q25_impact",
        "median_impact",
        "q75_impact",
        "q95_impact",
    ]
    if samples.empty:
        return pd.DataFrame(
            {
                "time_s": output_grid.astype(np.float64),
                "n_samples": np.zeros(output_grid.shape, dtype=np.int64),
                "mean_impact": np.nan,
                "std_impact": np.nan,
                "q05_impact": np.nan,
                "q25_impact": np.nan,
                "median_impact": np.nan,
                "q75_impact": np.nan,
                "q95_impact": np.nan,
            }
        )[columns]

    grouped = samples.groupby("time_s", sort=True)["impact"]
    summary = grouped.agg(
        n_samples="count",
        mean_impact="mean",
        std_impact="std",
        q05_impact=lambda x: float(np.quantile(x, 0.05)),
        q25_impact=lambda x: float(np.quantile(x, 0.25)),
        median_impact="median",
        q75_impact=lambda x: float(np.quantile(x, 0.75)),
        q95_impact=lambda x: float(np.quantile(x, 0.95)),
    ).reset_index()
    return summary[columns]


def _sample_previous_value(
    *,
    event_times: np.ndarray,
    event_values: np.ndarray,
    output_grid: np.ndarray,
    initial_value: float = 0.0,
) -> np.ndarray:
    """Sample a step path on `output_grid` using the latest event value."""
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


def _zero_sample_rows(
    episode_id: int,
    n_simulations: int,
    output_grid: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": int(episode_id),
            "simulation": int(sim_idx),
            "time_s": float(t),
            "impact": 0.0,
        }
        for sim_idx in range(int(n_simulations))
        for t in output_grid
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


def _output_grid(cfg: ImpactSeriesPipelineConfig) -> np.ndarray:
    n_steps = int(np.floor(cfg.horizon_seconds / cfg.output_step_seconds))
    grid = np.arange(n_steps + 1, dtype=np.float64) * float(cfg.output_step_seconds)
    if grid[-1] < cfg.horizon_seconds:
        grid = np.append(grid, float(cfg.horizon_seconds))
    return grid


def _timestamp_like(ts: pd.Series, value: object) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if getattr(ts.dt, "tz", None) is not None and out.tzinfo is None:
        out = out.tz_localize(ts.dt.tz)
    return out


def _empty_episodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": pd.Series(dtype=np.int64),
            "window_start": pd.Series(dtype=object),
            "first_source_row_pos": pd.Series(dtype=np.int64),
            "selected_source_rows": pd.Series(dtype=object),
            "n_selected_orders": pd.Series(dtype=np.int64),
            "post_span_s": pd.Series(dtype=np.float64),
            "selected_qty": pd.Series(dtype=np.int64),
        }
    )


def _empty_episode_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": pd.Series(dtype=np.int64),
            "order_slot": pd.Series(dtype=np.int64),
            "source_row_pos": pd.Series(dtype=np.int64),
            "post_ts": pd.Series(dtype=object),
            "post_time_s": pd.Series(dtype=np.float64),
            "qty": pd.Series(dtype=np.int64),
        }
    )


def _plot_impact_series(
    path: Path,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: ImpactSeriesPipelineConfig,
) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    if not samples.empty:
        for _, path_rows in samples.groupby(["episode_id", "simulation"], sort=False):
            path_rows = path_rows.sort_values("time_s")
            ax.plot(
                path_rows["time_s"],
                path_rows["impact"],
                color="0.45",
                linewidth=0.65,
                alpha=0.16,
                zorder=1,
            )
    if not summary.empty and summary["mean_impact"].notna().any():
        x = summary["time_s"].to_numpy(dtype=np.float64)
        mean = summary["mean_impact"].to_numpy(dtype=np.float64)
        q05 = summary["q05_impact"].to_numpy(dtype=np.float64)
        q95 = summary["q95_impact"].to_numpy(dtype=np.float64)
        ax.fill_between(
            x,
            q05,
            q95,
            color="#5b8cc0",
            alpha=0.18,
            linewidth=0,
            label="5-95%",
            zorder=2,
        )
        ax.plot(x, mean, color="#1f5d99", linewidth=2.0, label="mean", zorder=3)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5, zorder=0)
    ax.set_title(
        f"Price impact after {cfg.n_orders_per_episode} random passive L rows, "
        f"post={cfg.raw_side}, market={cfg.market_side}"
    )
    ax.set_xlabel("seconds from first selected limit")
    ax.set_ylabel("impact")
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
    except (ImportError, ModuleNotFoundError):
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        code_python = repo_root / "code" / "python"
        if str(code_python) not in sys.path:
            sys.path.insert(0, str(code_python))
        try:
            import simproj  # type: ignore
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "impact_series_pipeline requires the compiled simproj native "
                "extension. Build/install code/python before running the full "
                "pipeline."
            ) from exc

        return simproj


def _validate_config(cfg: ImpactSeriesPipelineConfig) -> None:
    validate_passive_impact_model_config(_passive_impact_model_config(cfg))
    if cfg.n_orders_per_episode <= 0:
        raise ValueError("n_orders_per_episode must be positive")
    if cfg.n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if cfg.horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if cfg.output_step_seconds <= 0:
        raise ValueError("output_step_seconds must be positive")
    if cfg.n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if cfg.post_span_seconds is not None and cfg.post_span_seconds < 0:
        raise ValueError("post_span_seconds must be nonnegative or None")
    if cfg.min_market_events < 0:
        raise ValueError("min_market_events must be nonnegative")
    if len(cfg.propagator_weights) != len(cfg.propagator_beta):
        raise ValueError("propagator_weights and propagator_beta must have matching lengths")
    if any(float(beta) <= 0.0 for beta in cfg.propagator_beta):
        raise ValueError("propagator_beta values must be positive")


def _passive_impact_model_config(cfg: ImpactSeriesPipelineConfig) -> PassiveImpactModelConfig:
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
    parser.add_argument("--aggregated-path", default=ImpactSeriesPipelineConfig.aggregated_path)
    parser.add_argument("--output-dir", default=ImpactSeriesPipelineConfig.output_dir)
    parser.add_argument("--n-orders-per-episode", type=int, default=ImpactSeriesPipelineConfig.n_orders_per_episode)
    parser.add_argument("--n-episodes", type=int, default=ImpactSeriesPipelineConfig.n_episodes)
    parser.add_argument("--post-span-seconds", type=float, default=ImpactSeriesPipelineConfig.post_span_seconds)
    parser.add_argument("--horizon-seconds", type=float, default=ImpactSeriesPipelineConfig.horizon_seconds)
    parser.add_argument("--output-step-seconds", type=float, default=ImpactSeriesPipelineConfig.output_step_seconds)
    parser.add_argument("--n-simulations", type=int, default=ImpactSeriesPipelineConfig.n_simulations)
    parser.add_argument("--raw-side", default=ImpactSeriesPipelineConfig.raw_side)
    parser.add_argument("--queue-col", default=ImpactSeriesPipelineConfig.queue_col)
    parser.add_argument("--market-side", default=ImpactSeriesPipelineConfig.market_side)
    parser.add_argument("--start-time", default=ImpactSeriesPipelineConfig.start_time)
    parser.add_argument("--end-time", default=ImpactSeriesPipelineConfig.end_time)
    parser.add_argument("--min-market-events", type=int, default=ImpactSeriesPipelineConfig.min_market_events)
    parser.add_argument("--seed", type=int, default=ImpactSeriesPipelineConfig.seed)
    parser.add_argument("--a-l", type=float, default=ImpactSeriesPipelineConfig.a_l)
    parser.add_argument("--b-l", type=float, default=ImpactSeriesPipelineConfig.b_l)
    parser.add_argument("--a-c", type=float, default=ImpactSeriesPipelineConfig.a_c)
    parser.add_argument("--b-c", type=float, default=ImpactSeriesPipelineConfig.b_c)
    parser.add_argument(
        "--impact-model",
        choices=["reduced_form", "tail_propagator", "propagator_tail", "structural"],
        default=ImpactSeriesPipelineConfig.impact_model,
    )
    parser.add_argument("--propagator-kappa", type=float, default=ImpactSeriesPipelineConfig.propagator_kappa)
    parser.add_argument("--propagator-gamma", type=float, default=ImpactSeriesPipelineConfig.propagator_gamma)
    parser.add_argument(
        "--propagator-weights",
        default=",".join(str(x) for x in ImpactSeriesPipelineConfig.propagator_weights),
    )
    parser.add_argument(
        "--propagator-beta",
        default=",".join(str(x) for x in ImpactSeriesPipelineConfig.propagator_beta),
    )
    parser.add_argument("--propagator-tail-zeta", type=float, default=ImpactSeriesPipelineConfig.propagator_tail_zeta)
    parser.add_argument("--c-kappa", type=float, default=ImpactSeriesPipelineConfig.c_kappa)
    parser.add_argument("--hawkes-mu", type=float, default=ImpactSeriesPipelineConfig.hawkes_mu)
    parser.add_argument("--hawkes-alpha", default=",".join(str(x) for x in ImpactSeriesPipelineConfig.hawkes_alpha))
    parser.add_argument("--hawkes-beta", default=",".join(str(x) for x in ImpactSeriesPipelineConfig.hawkes_beta))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ImpactSeriesPipelineConfig(
        **{
            **vars(args),
            "propagator_weights": _parse_floats(args.propagator_weights),
            "propagator_beta": _parse_floats(args.propagator_beta),
            "hawkes_alpha": _parse_floats(args.hawkes_alpha),
            "hawkes_beta": _parse_floats(args.hawkes_beta),
        }
    )
    summary = run_impact_series_pipeline(cfg)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2))


def _parse_floats(raw: str) -> tuple[float, ...]:
    if not raw:
        return ()
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
