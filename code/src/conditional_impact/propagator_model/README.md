# Propagator Model

Aggressive market impact computation based on kernel-weighted price response. We consider two price models that differ in how queue-dependent price sensitivity $\kappa$ interacts with the propagator kernel $G$: a **full propagator model** where every event is weighted by $\kappa(q)$ and propagated through $G$, and a **hybrid model** that separates the propagator (applied with a constant weight $\bar{\kappa}$) from the instantaneous queue-dependent correction. The hybrid decomposition was calibrated on data with most success by Gregoire Szymanski.

## Propagator Kernel

Both models share the same martingale-consistent propagator kernel. The condition $G'(t) = -G(0)\varphi(t)$ together with matching the expectation-based price $P_t = \lim_{T\to\infty} \mathbb{E}_t[N^a_T - N^b_T]$ gives $G(0) = 1/(1-\|\varphi\|_1)$ (mean cluster size). For $\varphi(t) = \sum_i \alpha_i e^{-\beta_i t}$:

$$\boxed{G(t) = \underbrace{1}_{\text{permanent}} + \sum_{i=1}^k \underbrace{\frac{\alpha_i / \beta_i}{1 - \|\varphi\|_1}}_{\text{weight}} \, e^{-\beta_i t}}$$

| Property | Value |
|----------|-------|
| $G(0)$ | $1/(1-\|\varphi\|_1)$ — mean cluster size, instantaneous overshoot |
| $G(\infty)$ | $1$ — permanent impact per event |
| Decay rates | $\beta_i$ (Hawkes kernel rates, **not** resolvent roots $\lambda_j$) |
| Inference | Directly from Hawkes parameters: $G(0) = 1/(1-\sum_i \alpha_i/\beta_i)$ |

---

## Price Model 1: Full Propagator

Every event passes through the propagator kernel with the full queue-dependent weight $\kappa(q)$:

$$P_t = P_0 + \int_0^t \kappa(q^a_s) \, G(t-s) \, dN^a_s - \int_0^t \kappa(q^b_s) \, G(t-s) \, dN^b_s$$

where $\kappa$ is any decreasing impact function and $dN^a, dN^b$ are independent Hawkes processes.

### Aggressive Impact (Full Propagator)

For a buy-side metaorder $N^{o,a}$ that depletes the ask queue from $q^a$ (baseline) to $\bar{q}^a$ (counterfactual with metaorder):

$$MI^a(t) = \int_0^t [\kappa(\bar{q}^a_s) - \kappa(q^a_s)] \, G(t-s) \, dN^a_s + \int_0^t \kappa(\bar{q}^a_s) \, G(t-s) \, dN^{o,a}_s$$

Both the queue-dependent correction (first term) and the metaorder contribution (second term) are propagated through $G$.

---

## Price Model 2: Hybrid

Decomposes the price into a propagator part (constant weight $\bar{\kappa}$) and an instantaneous queue-dependent residual:

$$P_t = P_0 + \bar{\kappa} \int_0^t G(t-s) \, d(N^a - N^b)_s + \int_0^t (\kappa(q^a_s) - \bar{\kappa}) \, dN^a_s - \int_0^t (\kappa(q^b_s) - \bar{\kappa}) \, dN^b_s$$

The first integral captures the temporal structure of market order clustering (via $G$) at a constant price-per-event $\bar{\kappa}$. The remaining integrals are instantaneous cumulative sums that correct for the deviation of $\kappa(q)$ from $\bar{\kappa}$ — they carry no temporal decay.

### Aggressive Impact (Hybrid)

For a buy-side metaorder $n$ (sell market orders consuming ask liquidity):

$$MI(t) = \bar{\kappa} \int_0^t G(t-s) \, dN^{o,a}_s + \int_0^t (\kappa(\bar{q}^a_s) - \kappa(q^a_s)) \, dN^a_s$$

| Term | Events | Weight | Temporal structure |
|------|--------|--------|--------------------|
| Propagator | Metaorder ($dN^{o,a}$) | Constant $\bar{\kappa}$ | Decays through $G(t-s)$ |
| Instantaneous | Market orders ($dN^a$) | $\kappa(\bar{q}) - \kappa(q)$ | No decay (cumulative sum) |

The propagator term is **deterministic** when the metaorder schedule is fixed: all simulation variance comes from the instantaneous term through the stochastic queue path $\bar{q}$.

---

## Efficient Computation

Both models share the same O(k)-per-event structure. Decompose $G$ into a permanent accumulator ($G(\infty) = 1$) and $k$ exponential states (weights $\frac{\alpha_i/\beta_i}{1-\|\varphi\|_1}$, rates $\beta_i$). At each event:

1. Decay exponential states by $e^{-\beta_i \Delta t}$
2. Compute contribution and route it:
   - **Full propagator**: all events update both the permanent accumulator and exponential states
   - **Hybrid**: metaorder events update the propagator (permanent + exponential states) with weight $\bar{\kappa}$; market order events update a separate non-decaying accumulator with weight $\kappa(\bar{q}) - \kappa(q)$
3. Impact = permanent + sum of exponential states (+ instant accumulator for hybrid)

## Usage

### Full Propagator

```rust
use simulation_project::conditional_impact::AggressiveImpactPath;
use simulation_project::models::MultiExponentialHawkes;

let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);

let c1 = 1000.0_f64;
let c2 = 0.01_f64;
let kappa = |q: f64| c1 * ((-c2 * q).exp() + 1.0_f64).ln().sqrt();

let impact = AggressiveImpactPath::from_queue_samples(
    &q_samples, &q_bar_samples, &eval_times,
    &is_market_order, &hawkes, &kappa,
);
```

### Hybrid

```rust
let c_kappa = 0.1_f64;
let kappa = |q: f64| -c_kappa * q;
let bar_kappa = 0.1;

let impact = AggressiveImpactPath::from_queue_samples_hybrid(
    &q_samples, &q_bar_samples, &eval_times,
    &is_market_order, &hawkes, &kappa, bar_kappa,
);
```
