"""Fill-before-next-reset diagnostics for post-reset first-level limits."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .execution_latency_grid import (
    CANCEL_DIM,
    LIMIT_DIM,
    MARKET_DIM,
    ExecutionLatencyGridConfig,
    _build_first_level_arrays,
    _load_raw_depth,
)


@dataclass(frozen=True)
class ResetIntervalFillConfig:
    """Measure whether post-reset first-level limit events fill before next reset."""

    raw_level_path: str = "experiments/impact_cost/load_experiments/data/raw/2025_05_29_ESM5.parquet"
    output_dir: str = "experiments/impact_cost/archive/diagnostics/data/reset_interval_fill_diagnostic"
    start_time: str | None = None
    end_time: str | None = None
    raw_side: str = "A"
    queue_col: str = "q_b"
    market_side: str | None = "B"
    cancellation_policy: str = "probabilistic_top"
    theta: float = 0.5
    max_rank_after_reset: int = 10
    seed: int = 2027
    cap_position_by_queue_post: bool = False


def run_reset_interval_fill_diagnostic(cfg: ResetIntervalFillConfig) -> dict[str, Any]:
    """Run the post-reset limit fill diagnostic and write CSV/JSON outputs."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_cfg = ExecutionLatencyGridConfig(
        raw_level_path=cfg.raw_level_path,
        start_time=cfg.start_time,
        end_time=cfg.end_time,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
        cancellation_policy=cfg.cancellation_policy,
        theta=cfg.theta,
        cap_position_by_queue_post=cfg.cap_position_by_queue_post,
        seed=cfg.seed,
    )
    raw = _filter_raw_time_bounds(_load_raw_depth(raw_cfg), cfg)
    arrays = _build_first_level_arrays(
        raw,
        raw_side=cfg.raw_side,
        queue_col=cfg.queue_col,
        market_side=cfg.market_side,
    )

    candidates = _post_reset_candidates(arrays, cfg)
    candidate_path = output_dir / "post_reset_limit_candidates.csv"
    summary_path = output_dir / "post_reset_fill_summary.csv"
    config_path = output_dir / "reset_interval_fill_config.json"
    candidates.to_csv(candidate_path, index=False)
    summary = _summarize_candidates(candidates)
    summary.to_csv(summary_path, index=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg) | {"q1_col": arrays["q1_col"]}, f, indent=2)

    return {
        "output_dir": output_dir,
        "candidate_path": candidate_path,
        "summary_path": summary_path,
        "config_path": config_path,
        "n_candidates": int(len(candidates)),
        "n_filled_before_next_reset": int(candidates["filled_before_next_reset"].sum())
        if len(candidates)
        else 0,
        "fill_rate": float(candidates["filled_before_next_reset"].mean())
        if len(candidates)
        else float("nan"),
    }


def _filter_raw_time_bounds(
    raw: pd.DataFrame,
    cfg: ResetIntervalFillConfig,
) -> pd.DataFrame:
    ts = raw["ts"]
    mask = np.ones(len(raw), dtype=bool)
    if cfg.start_time is not None:
        start = _timestamp_like(ts, cfg.start_time)
        mask &= (ts >= start).to_numpy()
    if cfg.end_time is not None:
        end = _timestamp_like(ts, cfg.end_time)
        mask &= (ts <= end).to_numpy()
    out = raw.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError("no raw rows remain after start/end filtering")
    return out


def _timestamp_like(ts: pd.Series, value: object) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if getattr(ts.dt, "tz", None) is not None and out.tzinfo is None:
        out = out.tz_localize(ts.dt.tz)
    return out


def _post_reset_candidates(
    arrays: dict[str, Any],
    cfg: ResetIntervalFillConfig,
) -> pd.DataFrame:
    reset_pos = np.flatnonzero(arrays["is_queue_reset"])
    limit_pos = np.flatnonzero((arrays["dims"] == LIMIT_DIM) & (arrays["qty"] > 0))
    if len(reset_pos) < 2 or len(limit_pos) == 0:
        return _empty_candidate_frame()

    rows: list[dict[str, Any]] = []
    limit_cursor = 0
    for interval_id, (start_pos, end_pos) in enumerate(zip(reset_pos[:-1], reset_pos[1:])):
        while limit_cursor < len(limit_pos) and limit_pos[limit_cursor] <= start_pos:
            limit_cursor += 1
        cursor = limit_cursor
        rank = 0
        while (
            cursor < len(limit_pos)
            and limit_pos[cursor] < end_pos
            and rank < int(cfg.max_rank_after_reset)
        ):
            candidate_pos = int(limit_pos[cursor])
            rank += 1
            fill = _simulate_single_order_until_reset(
                arrays,
                candidate_pos=candidate_pos,
                end_pos=int(end_pos),
                cfg=cfg,
                seed=int(cfg.seed) + interval_id * 1_000_003 + rank,
            )
            post_lag = float(arrays["times"][candidate_pos] - arrays["times"][start_pos])
            next_reset_lag = float(arrays["times"][end_pos] - arrays["times"][candidate_pos])
            rows.append(
                {
                    "interval_id": int(interval_id),
                    "rank_after_reset": int(rank),
                    "reset_source_row_pos": int(arrays["source_row_pos"][start_pos]),
                    "post_source_row_pos": int(arrays["source_row_pos"][candidate_pos]),
                    "next_reset_source_row_pos": int(arrays["source_row_pos"][end_pos]),
                    "post_lag_after_reset_s": post_lag,
                    "time_to_next_reset_from_post_s": next_reset_lag,
                    "post_qty": int(arrays["qty"][candidate_pos]),
                    "q1_post": float(arrays["q1_post"][candidate_pos]),
                    "filled_before_next_reset": bool(fill["filled"]),
                    "latency_s": fill["latency_s"],
                    "fill_source_row_pos": fill["fill_source_row_pos"],
                    "executed_qty": fill["executed_qty"],
                    "remaining_qty": fill["remaining_qty"],
                    "final_position_qty": fill["final_position_qty"],
                }
            )
            cursor += 1
    return pd.DataFrame(rows)


def _simulate_single_order_until_reset(
    arrays: dict[str, Any],
    *,
    candidate_pos: int,
    end_pos: int,
    cfg: ResetIntervalFillConfig,
    seed: int,
) -> dict[str, Any]:
    qty = arrays["qty"]
    dims = arrays["dims"]
    times = arrays["times"]
    q1_post = arrays["q1_post"]
    source_rows = arrays["source_row_pos"]

    initial_qty = int(qty[candidate_pos])
    remaining = initial_qty
    executed = 0
    position = max(float(q1_post[candidate_pos]), float(initial_qty))
    top_qty = 0
    rng = np.random.default_rng(seed)

    for row_pos in range(candidate_pos + 1, end_pos):
        event_qty = int(qty[row_pos])
        if event_qty <= 0:
            continue
        dim = int(dims[row_pos])
        if dim == LIMIT_DIM:
            top_qty += event_qty
        elif dim == MARKET_DIM:
            fill_qty = _market_fill_qty(position, remaining, event_qty)
            if fill_qty > 0:
                remaining -= fill_qty
                executed += fill_qty
                if remaining == 0:
                    return {
                        "filled": True,
                        "latency_s": float(times[row_pos] - times[candidate_pos]),
                        "fill_source_row_pos": int(source_rows[row_pos]),
                        "executed_qty": int(executed),
                        "remaining_qty": 0,
                        "final_position_qty": 0.0,
                    }
            position = 0.0 if remaining == 0 else max(position - event_qty, float(remaining))
        elif dim == CANCEL_DIM:
            desired_top_qty = _desired_top_cancel_qty(
                rng,
                event_qty,
                policy=cfg.cancellation_policy,
                theta=float(cfg.theta),
            )
            cancel_top_qty = min(top_qty, desired_top_qty)
            top_qty -= cancel_top_qty
            cancel_position_qty = event_qty - cancel_top_qty
            position = max(position - cancel_position_qty, float(remaining))

        if cfg.cap_position_by_queue_post:
            position = min(position, max(float(q1_post[row_pos]), 0.0))
            position = max(position, float(remaining))

    return {
        "filled": False,
        "latency_s": np.nan,
        "fill_source_row_pos": -1,
        "executed_qty": int(executed),
        "remaining_qty": int(remaining),
        "final_position_qty": float(position),
    }


def _market_fill_qty(position: float, remaining: int, event_qty: int) -> int:
    if remaining <= 0 or event_qty <= 0:
        return 0
    ahead_before = max(position - float(remaining), 0.0)
    consumed_until = min(position, float(event_qty))
    return int(min(max(np.floor(consumed_until - ahead_before), 0.0), float(remaining)))


def _desired_top_cancel_qty(
    rng: np.random.Generator,
    event_qty: int,
    *,
    policy: str,
    theta: float,
) -> int:
    policy = policy.lower()
    if policy == "top":
        return int(event_qty)
    if policy == "below":
        return 0
    if policy == "probabilistic_top":
        return int(rng.binomial(event_qty, theta))
    raise ValueError("cancellation_policy must be top, below, or probabilistic_top")


def _summarize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "n_candidates",
        "n_filled_before_next_reset",
        "fill_rate",
        "median_post_lag_after_reset_s",
        "median_time_to_next_reset_from_post_s",
        "median_latency_s",
        "median_q1_post",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    def add_summary(scope: str, frame: pd.DataFrame) -> None:
        filled = frame[frame["filled_before_next_reset"]]
        rows.append(
            {
                "scope": scope,
                "n_candidates": int(len(frame)),
                "n_filled_before_next_reset": int(len(filled)),
                "fill_rate": float(frame["filled_before_next_reset"].mean())
                if len(frame)
                else np.nan,
                "median_post_lag_after_reset_s": float(
                    frame["post_lag_after_reset_s"].median()
                )
                if len(frame)
                else np.nan,
                "median_time_to_next_reset_from_post_s": float(
                    frame["time_to_next_reset_from_post_s"].median()
                )
                if len(frame)
                else np.nan,
                "median_latency_s": float(filled["latency_s"].median())
                if len(filled)
                else np.nan,
                "median_q1_post": float(frame["q1_post"].median()) if len(frame) else np.nan,
            }
        )

    add_summary("rank<=max", candidates)
    for rank in (1, 2, 5, 10):
        rank_frame = candidates[candidates["rank_after_reset"] <= rank]
        if not rank_frame.empty:
            add_summary(f"rank<={rank}", rank_frame)
    for seconds in (0.001, 0.005, 0.01, 0.05, 0.1):
        early_frame = candidates[candidates["post_lag_after_reset_s"] <= seconds]
        add_summary(f"post_lag<={seconds:g}s", early_frame)
    for q1_threshold in (1, 2, 5, 10, 20):
        q1_frame = candidates[candidates["q1_post"] <= q1_threshold]
        add_summary(f"q1_post<={q1_threshold}", q1_frame)

    return pd.DataFrame(rows, columns=columns)


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interval_id": pd.Series(dtype=np.int64),
            "rank_after_reset": pd.Series(dtype=np.int64),
            "reset_source_row_pos": pd.Series(dtype=np.int64),
            "post_source_row_pos": pd.Series(dtype=np.int64),
            "next_reset_source_row_pos": pd.Series(dtype=np.int64),
            "post_lag_after_reset_s": pd.Series(dtype=np.float64),
            "time_to_next_reset_from_post_s": pd.Series(dtype=np.float64),
            "post_qty": pd.Series(dtype=np.int64),
            "q1_post": pd.Series(dtype=np.float64),
            "filled_before_next_reset": pd.Series(dtype=bool),
            "latency_s": pd.Series(dtype=np.float64),
            "fill_source_row_pos": pd.Series(dtype=np.int64),
            "executed_qty": pd.Series(dtype=np.int64),
            "remaining_qty": pd.Series(dtype=np.int64),
            "final_position_qty": pd.Series(dtype=np.float64),
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-level-path", default=ResetIntervalFillConfig.raw_level_path)
    parser.add_argument("--output-dir", default=ResetIntervalFillConfig.output_dir)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--raw-side", default=ResetIntervalFillConfig.raw_side)
    parser.add_argument("--queue-col", default=ResetIntervalFillConfig.queue_col)
    parser.add_argument("--market-side", default=ResetIntervalFillConfig.market_side)
    parser.add_argument(
        "--cancellation-policy",
        default=ResetIntervalFillConfig.cancellation_policy,
        choices=["top", "below", "probabilistic_top"],
    )
    parser.add_argument("--theta", type=float, default=ResetIntervalFillConfig.theta)
    parser.add_argument(
        "--max-rank-after-reset",
        type=int,
        default=ResetIntervalFillConfig.max_rank_after_reset,
    )
    parser.add_argument("--seed", type=int, default=ResetIntervalFillConfig.seed)
    parser.add_argument("--cap-position-by-queue-post", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_reset_interval_fill_diagnostic(
        ResetIntervalFillConfig(
            raw_level_path=args.raw_level_path,
            output_dir=args.output_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            raw_side=args.raw_side,
            queue_col=args.queue_col,
            market_side=args.market_side,
            cancellation_policy=args.cancellation_policy,
            theta=args.theta,
            max_rank_after_reset=args.max_rank_after_reset,
            seed=args.seed,
            cap_position_by_queue_post=args.cap_position_by_queue_post,
        )
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
