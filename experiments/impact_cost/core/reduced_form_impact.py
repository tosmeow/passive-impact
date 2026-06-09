"""Reduced-form price impact helpers for the impact-cost experiment.

These helpers implement the observable price approximation from Remark 2.4 of
the local reference draft. They deliberately use fitted price-propagator
coefficients directly instead of inverting them to Hawkes parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .level_execution import price_sign_for_queue


DEFAULT_PROPAGATOR_KAPPA: float = 0.00895780
DEFAULT_PROPAGATOR_GAMMA: float = -0.00001713
DEFAULT_PROPAGATOR_WEIGHTS: tuple[float, ...] = (
    -0.00102289,
    0.00084759,
    0.00161378,
    0.00031951,
)
DEFAULT_PROPAGATOR_BETA: tuple[float, ...] = (10.0, 1.0, 0.1, 0.01)


@dataclass(frozen=True)
class ReducedFormPropagator:
    """Fitted coefficients for the Remark-2.4 price approximation.

    The fitted price kernel is represented as

        g(t) = kappa_s + sum_i weights_i exp(-beta_i t).

    `kappa` stores `kappa_s`, the constant-sensitivity propagator level.
    `gamma` stores the reduced-form affine queue slope `kappa_1` in
    `kappa(q) = kappa_0 + kappa_1 q`; in configs this is
    `propagator_gamma`, not the structural Hawkes parameter `c_kappa`.
    For passive queue perturbations with unchanged signed market-order flow,
    the propagator term cancels and only the queue-weighted `H_t` term remains.
    """

    kappa: float = DEFAULT_PROPAGATOR_KAPPA
    gamma: float = DEFAULT_PROPAGATOR_GAMMA
    weights: tuple[float, ...] = DEFAULT_PROPAGATOR_WEIGHTS
    beta: tuple[float, ...] = DEFAULT_PROPAGATOR_BETA

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.beta):
            raise ValueError("weights and beta must have matching lengths")
        if any(b <= 0.0 for b in self.beta):
            raise ValueError("all beta values must be positive")


def passive_reduced_form_impact_from_queue_samples(
    q_samples: np.ndarray,
    q_bar_samples: np.ndarray,
    *,
    queue_col: str,
    gamma: float = DEFAULT_PROPAGATOR_GAMMA,
) -> np.ndarray:
    """Return passive reduced-form impact at consuming market times.

    `q_bar_samples` is the factual/with-us queue and `q_samples` is the no-us
    baseline queue, both sampled immediately after the same consuming-side
    market events. Since passive limit insertion leaves the signed market-order
    path common, the propagator part of Remark 2.4 cancels in the price
    difference. The remaining queue term is

        kappa_1 * int (q_bar - q) dN_a

    for ask queues, and with the opposite sign for bid queues. The `gamma`
    argument is this reduced-form `kappa_1` coefficient.
    """
    q = np.asarray(q_samples, dtype=np.float64)
    q_bar = np.asarray(q_bar_samples, dtype=np.float64)
    if q.shape != q_bar.shape:
        raise ValueError("q_samples and q_bar_samples must have matching shapes")

    signed_queue_diff = price_sign_for_queue(queue_col) * (q_bar - q)
    return float(gamma) * np.cumsum(signed_queue_diff)


def effective_passive_kernel_coefficients(
    *,
    coefficients: ReducedFormPropagator = ReducedFormPropagator(),
    c_lambda: float,
) -> np.ndarray:
    """Return propagator-input effective kernel coefficients.

    If the fitted price propagator is

        g(t) = kappa_s + sum_i w_i exp(-beta_i t),

    the passive tail formula uses the dimensionless effective kernel

        K_C(t) = sum_i eta_i exp(-beta_i t),
        eta_i = beta_i w_i / (kappa_s * (beta_i + C_lambda)).

    This avoids interpreting signed propagator weights as Hawkes amplitudes.
    """
    kappa = float(coefficients.kappa)
    if kappa == 0.0:
        raise ValueError("propagator kappa must be nonzero")
    c_lbd = float(c_lambda)
    if c_lbd <= 0.0:
        raise ValueError("c_lambda must be positive")
    beta = np.asarray(coefficients.beta, dtype=np.float64)
    weights = np.asarray(coefficients.weights, dtype=np.float64)
    if beta.shape != weights.shape:
        raise ValueError("weights and beta must have matching lengths")
    if np.any(beta <= 0.0):
        raise ValueError("all beta values must be positive")
    return beta * weights / (kappa * (beta + c_lbd))


def passive_propagator_tail_impact_from_queue_samples(
    q_samples: np.ndarray,
    q_bar_samples: np.ndarray,
    market_times: np.ndarray,
    *,
    queue_col: str,
    coefficients: ReducedFormPropagator = ReducedFormPropagator(),
    c_lambda: float,
    queue_sensitivity: float | None = None,
    zeta: float = 0.0,
) -> np.ndarray:
    """Return the Rust-style passive impact path using propagator-input tails.

    The output is aligned with `market_times`. At each consuming market event we
    decay the exponential states, add the current event's effective-kernel
    coefficients, then evaluate

        queue_sensitivity * (sum_s signed_U_s + signed_U_t * F_t),

    where `signed_U = price_sign_for_queue(queue_col) * (q_bar - q)` and
    `F_t = zeta + sum_i state_i(t)`. `queue_sensitivity` is the reduced-form
    queue slope `kappa_1`; if omitted, `coefficients.gamma` is used.
    """
    q = np.asarray(q_samples, dtype=np.float64)
    q_bar = np.asarray(q_bar_samples, dtype=np.float64)
    times = np.asarray(market_times, dtype=np.float64)
    if q.shape != q_bar.shape or q.shape != times.shape:
        raise ValueError("q_samples, q_bar_samples, and market_times must match")
    if times.ndim != 1:
        raise ValueError("market_times must be one-dimensional")
    if times.size and np.any(np.diff(times) < 0.0):
        raise ValueError("market_times must be sorted")

    eta = effective_passive_kernel_coefficients(
        coefficients=coefficients,
        c_lambda=float(c_lambda),
    )
    beta = np.asarray(coefficients.beta, dtype=np.float64)
    states = np.zeros(beta.shape, dtype=np.float64)
    signed_queue_diff = price_sign_for_queue(queue_col) * (q_bar - q)
    out = np.zeros(times.shape, dtype=np.float64)
    sensitivity = float(
        coefficients.gamma if queue_sensitivity is None else queue_sensitivity
    )

    cumulative = 0.0
    prev_t = 0.0
    for idx, (t, diff) in enumerate(zip(times, signed_queue_diff)):
        dt = float(t) - prev_t
        if dt < 0.0:
            raise ValueError("market_times must be sorted")
        states *= np.exp(-beta * dt)
        states += eta
        forecast = float(zeta) + float(states.sum())
        cumulative += float(diff)
        out[idx] = sensitivity * (cumulative + float(diff) * forecast)
        prev_t = float(t)
    return out


def propagator_impact_from_events(
    eval_times: np.ndarray,
    event_times: np.ndarray,
    *,
    coefficients: ReducedFormPropagator = ReducedFormPropagator(),
    event_sign: float = 1.0,
) -> np.ndarray:
    """Evaluate direct signed-flow propagator impact on an output grid.

    This is the direct Remark-2.4 propagator component for extra signed-flow
    events, such as aggressive metaorder trades:

        sum_s sign * (kappa + sum_i w_i exp(-beta_i (t-s))) 1_{s <= t}.

    Passive limit insertions usually do not need this term because their signed
    market-order flow is common between factual and no-us worlds.
    """
    grid = np.asarray(eval_times, dtype=np.float64)
    events = np.asarray(event_times, dtype=np.float64)
    if grid.ndim != 1 or events.ndim != 1:
        raise ValueError("eval_times and event_times must be one-dimensional")
    if events.size == 0:
        return np.zeros(grid.shape, dtype=np.float64)

    out = np.zeros(grid.shape, dtype=np.float64)
    for idx, t in enumerate(grid):
        ages = t - events[events <= t]
        if ages.size == 0:
            continue
        transient = 0.0
        for weight, beta in zip(coefficients.weights, coefficients.beta):
            transient += float(weight) * np.exp(-float(beta) * ages).sum()
        out[idx] = float(event_sign) * (
            float(coefficients.kappa) * ages.size + transient
        )
    return out
