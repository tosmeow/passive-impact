# Hybrid Propagator Model

Aggressive market impact computation based on a hybrid price response. The
model separates the temporal clustering of metaorder flow from the
queue-dependent correction caused by ordinary market orders:

$$
P_t = P_0 + \bar{\kappa}\int_0^t G(t-s)\,d(N^a-N^b)_s
    + \int_0^t(\kappa(q^a_s)-\bar{\kappa})\,dN^a_s
    - \int_0^t(\kappa(q^b_s)-\bar{\kappa})\,dN^b_s.
$$

## Propagator Kernel

The model uses the martingale-consistent propagator kernel. The condition
$G'(t) = -G(0)\varphi(t)$ together with matching the expectation-based price
$P_t = \lim_{T\to\infty} \mathbb{E}_t[N^a_T - N^b_T]$ gives
$G(0) = 1/(1-\|\varphi\|_1)$ (mean cluster size). For
$\varphi(t) = \sum_i \alpha_i e^{-\beta_i t}$:

$$
\boxed{G(t) = 1 + \sum_{i=1}^k
\frac{\alpha_i / \beta_i}{1 - \|\varphi\|_1} e^{-\beta_i t}}
$$

| Property | Value |
|----------|-------|
| $G(0)$ | $1/(1-\|\varphi\|_1)$ — mean cluster size, instantaneous overshoot |
| $G(\infty)$ | $1$ — permanent impact per event |
| Decay rates | $\beta_i$ (Hawkes kernel rates, not resolvent roots $\lambda_j$) |
| Inference | Directly from Hawkes parameters: $G(0) = 1/(1-\sum_i \alpha_i/\beta_i)$ |

## Aggressive Impact

For a buy-side metaorder $N^{o,a}$ that depletes the ask queue from $q^a$
(baseline) to $\bar{q}^a$ (counterfactual with metaorder):

$$
MI_t = \bar{\kappa}\int_0^t G(t-s)\,dN^{o,a}_s
     + \int_0^t [\kappa(\bar{q}^a_s)-\kappa(q^a_s)]\,dN^a_s.
$$

| Term | Events | Weight | Temporal structure |
|------|--------|--------|--------------------|
| Propagator | Metaorder ($dN^{o,a}$) | Constant $\bar{\kappa}$ | Decays through $G(t-s)$ |
| Instantaneous | Market orders ($dN^a$) | $\kappa(\bar{q}) - \kappa(q)$ | No decay (cumulative sum) |

The propagator term is deterministic when the metaorder schedule is fixed; all
conditional-simulation variance comes from the instantaneous term through the
stochastic queue path.

## Efficient Computation

The implementation is O(k) per event:

1. Decay exponential propagator states by $e^{-\beta_i \Delta t}$.
2. At metaorder events, update the permanent and exponential propagator states
   with weight $\bar{\kappa}$.
3. At ordinary market-order events, update a separate non-decaying accumulator
   with weight $\kappa(\bar{q}) - \kappa(q)$.
4. Impact = permanent propagator accumulator + exponential states +
   instantaneous accumulator.

## Usage

```rust
use simulation_project::conditional_impact::AggressiveImpactPath;
use simulation_project::models::MultiExponentialHawkes;

let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);

let c_kappa = 0.001_f64;
let kappa = |q: f64| -c_kappa * q;
let bar_kappa = 0.01_f64;

let impact = AggressiveImpactPath::from_queue_samples(
    &q_samples,
    &q_bar_samples,
    &eval_times,
    &is_market_order,
    &hawkes,
    &kappa,
    bar_kappa,
);
```
