"""Shared dataframe helpers for impact-cost experiments."""
from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from .cost_utils import LIMIT, event_seconds


AGGREGATED_DEPTH_COLUMNS = ("ts", "order_type", "side", "qty", "q_b", "q_a")


def load_aggregated_depth(path: str | Path) -> pd.DataFrame:
    """Load the processed factual depth input used by lifecycle experiments."""
    input_path = Path(path)
    columns = list(AGGREGATED_DEPTH_COLUMNS)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, usecols=columns)
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df = pd.read_parquet(input_path, columns=columns)
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["source_row_pos"] = np.arange(len(df), dtype=np.int64)
    return df


def window_from_aggregated(
    aggregated: pd.DataFrame,
    start: pd.Timestamp,
    *,
    horizon_seconds: float,
    warmup_seconds: float = 0.0,
) -> pd.DataFrame:
    """Return a time window from processed factual depth rows."""
    ts = aggregated["ts"]
    if getattr(ts.dt, "tz", None) is not None and start.tzinfo is None:
        start = start.tz_localize(ts.dt.tz)
    warm_start = start - pd.Timedelta(seconds=float(warmup_seconds))
    end = start + pd.Timedelta(seconds=float(horizon_seconds))
    return aggregated[(ts >= warm_start) & (ts <= end)].copy().reset_index(drop=True)


def timestamp_like(ts: pd.Series, value: object) -> pd.Timestamp:
    """Parse `value` as a timestamp with the same timezone convention as `ts`."""
    out = pd.Timestamp(value)
    if getattr(ts.dt, "tz", None) is not None and out.tzinfo is None:
        out = out.tz_localize(ts.dt.tz)
    return out


def candidate_episodes(aggregated: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Select candidate lifecycle windows from processed factual limit rows."""
    types = aggregated["order_type"].astype(str).str.lower()
    sides = aggregated["side"].astype(str)
    limit_rows = aggregated[(types == LIMIT) & (sides == cfg.raw_side)][
        ["source_row_pos", "ts", "qty"]
    ].copy()
    if limit_rows.empty:
        return empty_candidate_episodes()

    origin = aggregated["ts"].iloc[0]
    limit_seconds = event_seconds(limit_rows, origin=origin)
    max_start_s = float((aggregated["ts"].iloc[-1] - origin).total_seconds()) - float(
        cfg.horizon_seconds
    )
    keep = limit_seconds <= max_start_s
    if cfg.start_time is not None:
        start_bound = timestamp_like(aggregated["ts"], cfg.start_time)
        keep &= (limit_rows["ts"] >= start_bound).to_numpy()
    if cfg.end_time is not None:
        end_bound = timestamp_like(aggregated["ts"], cfg.end_time)
        keep &= (limit_rows["ts"] <= end_bound).to_numpy()
    limit_rows = limit_rows[keep].copy()
    limit_seconds = limit_seconds[keep]
    if limit_rows.empty:
        return empty_candidate_episodes()

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
            idx = np.sort(
                rng.choice(
                    np.arange(len(selected)),
                    size=int(cfg.max_episodes),
                    replace=False,
                )
            )
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


def empty_candidate_episodes() -> pd.DataFrame:
    """Return the empty candidate-episode schema."""
    return pd.DataFrame(
        {
            "episode_id": pd.Series(dtype=np.int64),
            "window_start": pd.Series(dtype=object),
            "first_source_row_pos": pd.Series(dtype=np.int64),
        }
    )


def select_limit_sequences(
    aggregated: pd.DataFrame,
    cfg: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly select factual limit-row sequences for archived diagnostics."""
    types = aggregated["order_type"].astype(str).str.lower()
    sides = aggregated["side"].astype(str)
    limit_rows = aggregated[(types == LIMIT) & (sides == cfg.raw_side)][
        ["source_row_pos", "ts", "qty"]
    ].copy()
    limit_rows = limit_rows.reset_index(drop=True)
    if limit_rows.empty:
        return empty_sequence_episodes(), empty_sequence_orders()

    n = int(cfg.n_orders_per_episode)
    if len(limit_rows) < n:
        return empty_sequence_episodes(), empty_sequence_orders()

    origin = aggregated["ts"].iloc[0]
    limit_seconds = event_seconds(limit_rows, origin=origin)
    m = len(limit_rows) - n + 1
    start_seconds = limit_seconds[:m]
    end_sequence_seconds = limit_seconds[n - 1 :]
    max_data_seconds = float((aggregated["ts"].iloc[-1] - origin).total_seconds())

    candidates = start_seconds + float(cfg.horizon_seconds) <= max_data_seconds
    if cfg.post_span_seconds is not None:
        candidates &= (end_sequence_seconds - start_seconds) <= float(cfg.post_span_seconds)

    if cfg.start_time is not None:
        start_bound = timestamp_like(limit_rows["ts"], cfg.start_time)
        candidates &= (limit_rows["ts"].iloc[:m] >= start_bound).to_numpy()
    if cfg.end_time is not None:
        end_bound = timestamp_like(limit_rows["ts"], cfg.end_time)
        candidates &= (limit_rows["ts"].iloc[n - 1 :] <= end_bound).to_numpy()

    candidate_offsets = np.flatnonzero(candidates)
    if candidate_offsets.size == 0:
        return empty_sequence_episodes(), empty_sequence_orders()

    rng = np.random.default_rng(cfg.seed)
    n_selected = min(int(cfg.n_episodes), int(candidate_offsets.size))
    chosen_offsets = rng.choice(candidate_offsets, size=n_selected, replace=False)

    episode_rows = []
    order_rows = []
    for episode_id, offset in enumerate(chosen_offsets):
        seq = limit_rows.iloc[offset : offset + n].copy()
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
                "post_span_s": float(seq_seconds[-1] - seq_seconds[0])
                if len(seq_seconds)
                else 0.0,
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


def empty_sequence_episodes() -> pd.DataFrame:
    """Return the empty selected-sequence episode schema."""
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


def empty_sequence_orders() -> pd.DataFrame:
    """Return the empty selected-sequence order schema."""
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


def sample_previous_value(
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


def queue_to_u32(values: np.ndarray) -> np.ndarray:
    """Round nonnegative queue samples into the native unsigned integer format."""
    return np.rint(np.maximum(np.asarray(values, dtype=np.float64), 0.0)).astype(
        np.uint32
    )
