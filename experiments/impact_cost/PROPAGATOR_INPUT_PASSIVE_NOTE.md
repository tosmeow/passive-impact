# Propagator-Input Passive Impact Note

This note records the passive-impact formulation to use when the calibrated
input is a fitted price propagator exponential sum, not an underlying Hawkes
kernel.

## Setup

Let

```text
U_t = q_bar_t - q_t
```

be the factual/no-us queue displacement. For the affine queue drift, use the
code sign convention

```text
C_lambda = b_c - b_l > 0
```

which corresponds to the paper's `c_lambda = b_l - b_c < 0`. The queue
displacement continuation is discounted by `exp(-C_lambda r)`.

Assume the fitted constant-sensitivity price propagator is available directly:

```text
g(u) = kappa_s + sum_i w_i exp(-beta_i u)
     = kappa_s * xi(u),

xi(u) = 1 + sum_i a_i exp(-beta_i u),
a_i = w_i / kappa_s.
```

Here `kappa_s = g(infinity)` is the permanent price contribution per market
event. The fitted transient weights `w_i` may have either sign. They should not
be interpreted as Hawkes excitation amplitudes.

## From Propagator To Discounted Forecast Kernel

For the constant-sensitivity expectation formula, the dimensionless propagator
can be written as

```text
xi(a) = 1 + int_a^infinity r(v) dv,
```

where `r(v)` is the history-to-future-intensity response kernel implied by the
propagator. Therefore

```text
r(a) = - xi'(a)
     = sum_i beta_i a_i exp(-beta_i a).
```

The passive continuation term needs the discounted future intensity forecast:

```text
F_t = int_0^infinity exp(-C_lambda r)
        E_t[lambda_{t+r}] dr.
```

Using the propagator-implied response kernel,

```text
F_t = zeta + int_0^t K_C(t-s) dN_s,

K_C(a) = int_0^infinity exp(-C_lambda r) r(a+r) dr
       = sum_i eta_i exp(-beta_i a),

eta_i = beta_i a_i / (beta_i + C_lambda)
      = beta_i w_i / (kappa_s * (beta_i + C_lambda)).
```

Thus the effective passive kernel is obtained directly from the fitted
propagator by scaling each exponential component by

```text
beta_i / (beta_i + C_lambda).
```

This is the propagator-input analogue of the paper's Theorem 4.5 / Remark 4.6,
where the effective kernel is written as `sum_i gamma_i exp(-beta_i u)`. In
the propagator-input case, those coefficients are the `eta_i` above, not
Hawkes `alpha_i`.

## Passive Impact Formula

With reduced-form queue sensitivity

```text
kappa(q) = kappa_0 + kappa_1 q,
kappa_1 = propagator_gamma,
```

the passive price impact path is

```text
MI_t =
  kappa_1 * int_0^t U_s dN_s
  + kappa_1 * U_t * (zeta + int_0^t K_C(t-s) dN_s).
```

The first term is the realized market-event contribution. The second term is
the continuation value of the current queue displacement. This second term can
decay after posting stops as `U_t` reconverges and as the exponential states
decay, even though the realized cumulative term can keep increasing while
`U_s` remains nonzero at market events.

In the structural Hawkes formula from the paper, the analogous queue slope is
`c_kappa`. In the `tail_propagator` implementation we intentionally use the
fitted reduced-form slope `propagator_gamma = kappa_1` instead; `c_kappa` is
reserved for `impact_model="structural"`.

`zeta` is the baseline future-arrival contribution. It is not identified by
the propagator shape alone. A practical choice is to set

```text
zeta = market_event_rate / C_lambda
```

using the empirical consuming-side market-event rate, or to fit/calibrate it
separately.

## Image Calibration Example

Using the image-calibrated propagator

```text
kappa_s = 0.00895780
(w_i, beta_i) =
  (-0.00102289, 10.0),
  ( 0.00084759,  1.0),
  ( 0.00161378,  0.1),
  ( 0.00031951,  0.01),
```

the dimensionless transient propagator coefficients are

```text
a_i = w_i / kappa_s
    = (-0.1141898680, 0.0946203309, 0.1801536091, 0.0356683561).
```

For unscaled queue slopes

```text
b_l = -0.000097, b_c = 0.0000989, C_lambda = 0.0001959,
```

the effective passive-kernel coefficients are

```text
eta_i =
  (-0.1141876311, 0.0946017984, 0.1798013782, 0.0349830384).
```

For the older x100 queue slopes

```text
b_l = -0.0097, b_c = 0.00989, C_lambda = 0.01959,
```

the effective passive-kernel coefficients are

```text
eta_i =
  (-0.1139666075, 0.0928023332, 0.1506427035, 0.0120541927).
```

The slow `beta=0.01` component is much more suppressed in the x100 case because
`beta / (beta + C_lambda)` is then about `0.338`.

## Implementation Hook

The Python impact-cost pipelines expose this formulation as
`--impact-model tail_propagator`; `propagator_tail` is kept as a compatibility
alias. The implementation evaluates the path at consuming market-event times,
mirroring the old Rust single-queue helper:

```text
state_i(t_j) = exp(-beta_i (t_j - t_{j-1})) state_i(t_{j-1}) + eta_i
F_j = zeta + sum_i state_i(t_j)
MI_j = kappa_1 * (sum_{m <= j} U_m + U_j F_j)
```

The code uses the signed single-queue convention
`U_j = price_sign_for_queue(queue_col) * (q_bar_j - q_j)` and uses the fitted
`propagator_gamma` as the reduced-form queue slope `kappa_1`. The configurable
baseline term is `propagator_tail_zeta`; it defaults to `0.0`.
