"""Passive impact path and execution-cost accounting.

This module is the narrow bridge from simulated queue paths to the final
objects used by the impact-cost experiment:

- a market-time impact path;
- execution-time cost jumps;
- total passive execution cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .cost_utils import cost_from_fills
from .level_execution import price_sign_for_queue
from .reduced_form_impact import (
    DEFAULT_PROPAGATOR_BETA,
    DEFAULT_PROPAGATOR_GAMMA,
    DEFAULT_PROPAGATOR_KAPPA,
    DEFAULT_PROPAGATOR_WEIGHTS,
    ReducedFormPropagator,
    passive_reduced_form_impact_from_queue_samples,
    passive_propagator_tail_impact_from_queue_samples,
)


TAIL_PROPAGATOR_IMPACT_MODELS = {"propagator_tail", "tail_propagator"}


@dataclass(frozen=True)
class PassiveImpactModelConfig:
    """Configuration for converting queue samples into passive price impact.

    Notation mapping:

    - `propagator_kappa` is the constant price-propagator level `kappa_s`.
    - `propagator_gamma` is the reduced-form queue slope `kappa_1`; the
      ask/bid sign is applied with `price_sign_for_queue`.
    - `c_kappa` and `hawkes_*` are structural Hawkes-model parameters and are
      only used by `impact_model="structural"`.
    """

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
    b_l: float = -0.0097
    b_c: float = 0.00989


def validate_passive_impact_model_config(cfg: PassiveImpactModelConfig) -> None:
    """Validate the model-choice fields shared by impact-cost workflows."""
    if cfg.impact_model not in {
        "reduced_form",
        "propagator_tail",
        "tail_propagator",
        "structural",
    }:
        raise ValueError(
            "impact_model must be 'reduced_form', 'tail_propagator', "
            "'propagator_tail', or 'structural'"
        )
    if cfg.impact_model in TAIL_PROPAGATOR_IMPACT_MODELS:
        if len(cfg.propagator_weights) != len(cfg.propagator_beta):
            raise ValueError(
                "propagator_weights and propagator_beta must have matching lengths"
            )
        if any(float(beta) <= 0.0 for beta in cfg.propagator_beta):
            raise ValueError("propagator_beta values must be positive")
        if float(cfg.propagator_kappa) == 0.0:
            raise ValueError("propagator_kappa must be nonzero")
        if float(cfg.b_c) - float(cfg.b_l) <= 0.0:
            raise ValueError("b_c - b_l must be positive for tail_propagator")
    if cfg.impact_model == "structural":
        if len(cfg.hawkes_alpha) != len(cfg.hawkes_beta):
            raise ValueError("hawkes_alpha and hawkes_beta must have matching lengths")
        if any(float(beta) <= 0.0 for beta in cfg.hawkes_beta):
            raise ValueError("hawkes_beta values must be positive")


def passive_impact_path_from_queue_samples(
    q_samples: np.ndarray,
    q_bar_samples: np.ndarray,
    market_times: np.ndarray,
    *,
    queue_col: str,
    cfg: PassiveImpactModelConfig,
    simproj: Any | None = None,
) -> np.ndarray:
    """Compute the passive impact path at consuming-side market times.

    `q_bar_samples` is the observed factual/with-us queue. `q_samples` is the
    simulated no-us queue. The returned array is aligned with `market_times`.
    """
    validate_passive_impact_model_config(cfg)
    q = np.asarray(q_samples, dtype=np.uint32)
    q_bar = np.asarray(q_bar_samples, dtype=np.uint32)
    times = np.asarray(market_times, dtype=np.float64)
    if q.shape != q_bar.shape or q.shape != times.shape:
        raise ValueError("q_samples, q_bar_samples, and market_times must match")

    if cfg.impact_model == "reduced_form":
        return passive_reduced_form_impact_from_queue_samples(
            q,
            q_bar,
            queue_col=queue_col,
            gamma=float(cfg.propagator_gamma),
        )

    if cfg.impact_model in TAIL_PROPAGATOR_IMPACT_MODELS:
        native_tail = (
            None
            if simproj is None
            else getattr(simproj, "passive_tail_propagator_impact_from_queue_samples", None)
        )
        if native_tail is not None:
            queue_contribution = np.asarray(
                native_tail(
                    q,
                    q_bar,
                    times,
                    float(cfg.propagator_kappa),
                    [float(x) for x in cfg.propagator_weights],
                    [float(x) for x in cfg.propagator_beta],
                    float(cfg.b_l),
                    float(cfg.b_c),
                    float(cfg.propagator_gamma),
                    float(cfg.propagator_tail_zeta),
                ),
                dtype=np.float64,
            )
            # Native returns the one-queue contribution scaled by `kappa_1`;
            # this applies the ask/bid sign convention for price impact.
            return price_sign_for_queue(queue_col) * queue_contribution

        return passive_propagator_tail_impact_from_queue_samples(
            q,
            q_bar,
            times,
            queue_col=queue_col,
            coefficients=ReducedFormPropagator(
                kappa=float(cfg.propagator_kappa),
                gamma=float(cfg.propagator_gamma),
                weights=tuple(float(x) for x in cfg.propagator_weights),
                beta=tuple(float(x) for x in cfg.propagator_beta),
            ),
            c_lambda=float(cfg.b_c) - float(cfg.b_l),
            zeta=float(cfg.propagator_tail_zeta),
        )

    if simproj is None:
        raise ValueError("simproj is required for structural passive impact")

    queue_contribution = np.asarray(
        simproj.passive_flow_impact_from_queue_samples(
            q,
            q_bar,
            times,
            float(cfg.hawkes_mu),
            [float(x) for x in cfg.hawkes_alpha],
            [float(x) for x in cfg.hawkes_beta],
            float(cfg.b_l),
            float(cfg.b_c),
            float(cfg.c_kappa),
        ),
        dtype=np.float64,
    )
    return price_sign_for_queue(queue_col) * queue_contribution


def execution_cost_jump_series(
    fills: pd.DataFrame,
    *,
    market_times: np.ndarray,
    impact: np.ndarray,
    time_col: str = "fill_time_s",
    qty_col: str = "qty",
    copy_columns: tuple[str, ...] = (),
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return fill-time cost jumps and cumulative running cost.

    The impact value is sampled as a left limit, matching the passive execution
    convention used by `cost_from_fills`.
    """
    columns = [
        *(extra_columns or {}).keys(),
        *copy_columns,
        time_col,
        qty_col,
        "impact_left",
        "cost_jump",
        "contribution",
        "cumulative_cost",
    ]
    if fills.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})

    times = np.asarray(market_times, dtype=np.float64)
    values = np.asarray(impact, dtype=np.float64)
    if times.shape != values.shape:
        raise ValueError("market_times and impact must have matching shapes")

    fill_times = fills[time_col].to_numpy(dtype=np.float64)
    idx = np.searchsorted(times, fill_times, side="left") - 1

    rows: list[dict[str, Any]] = []
    running = 0.0
    for fill_row, impact_idx in zip(fills.itertuples(index=False), idx):
        qty = int(getattr(fill_row, qty_col))
        impact_left = float(values[impact_idx]) if impact_idx >= 0 else 0.0
        cost_jump = impact_left * qty
        running += cost_jump
        base = {} if extra_columns is None else dict(extra_columns)
        for col in copy_columns:
            value = getattr(fill_row, col)
            if isinstance(value, np.generic):
                value = value.item()
            base[col] = value
        base.update(
            {
                time_col: float(getattr(fill_row, time_col)),
                qty_col: qty,
                "impact_left": impact_left,
                "cost_jump": cost_jump,
                "contribution": cost_jump,
                "cumulative_cost": running,
            }
        )
        rows.append(base)
    return pd.DataFrame(rows, columns=columns)


def passive_cost_from_fills(
    fills: pd.DataFrame,
    *,
    market_times: np.ndarray,
    impact: np.ndarray,
    time_col: str = "fill_time_s",
    qty_col: str = "qty",
) -> float:
    """Return total passive cost from fill labels and a market-time impact path."""
    return float(
        cost_from_fills(
            fills,
            np.asarray(market_times, dtype=np.float64),
            np.asarray(impact, dtype=np.float64),
            time_col=time_col,
            qty_col=qty_col,
        )
    )
