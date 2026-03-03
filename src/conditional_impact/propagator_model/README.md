# Propagator Model

Aggressive market impact computation based on kernel-weighted price response.

## Price Model

Following eq. (4.6) of the reference paper:

$$P_t = P_0 + \int_0^t \kappa(q^a_s) \, G(t-s) \, dN^a_s - \int_0^t \kappa(q^b_s) \, G(t-s) \, dN^b_s$$

where $\kappa(q) = c_\kappa q + d_\kappa$, $dN^a, dN^b$ are independent Hawkes processes with kernel $\varphi$, and $G$ is the propagator kernel.

## Propagator Kernel

The martingale condition $G'(t) = -G(0)\varphi(t)$ together with matching the expectation-based price $P_t = \lim_{T\to\infty} \mathbb{E}_t[N^a_T - N^b_T]$ gives $G(0) = 1/(1-\|\varphi\|_1)$ (mean cluster size). For $\varphi(t) = \sum_i \alpha_i e^{-\beta_i t}$:

$$\boxed{G(t) = \underbrace{1}_{\text{permanent}} + \sum_{i=1}^k \underbrace{\frac{\alpha_i / \beta_i}{1 - \|\varphi\|_1}}_{\text{weight}} \, e^{-\beta_i t}}$$

| Property | Value |
|----------|-------|
| $G(0)$ | $1/(1-\|\varphi\|_1)$ — mean cluster size, instantaneous overshoot |
| $G(\infty)$ | $1$ — permanent impact per event |
| Decay rates | $\beta_i$ (Hawkes kernel rates, **not** resolvent roots $\lambda_j$) |
| Inference | Directly from Hawkes parameters: $G(0) = 1/(1-\sum_i \alpha_i/\beta_i)$ |

## Aggressive Impact Formula

Consider a buy-side metaorder that depletes the ask queue from $\bar{q}^a$ (counterfactual) to $q^a$ (realized). The market impact on the ask side at time $t$:

$$MI^a(t) = \int_0^t [\kappa(\bar{q}^a_s) - \kappa(q^a_s)] G(t-s) \, dN^a_s + \int_0^t \kappa(\bar{q}^a_s) G(t-s) \, dN^{o,a}_s$$

where $dN^a$ is the ordinary ask-side market order stream and $dN^{o,a}$ the metaorder stream (both consuming ask liquidity). The symmetric formula holds on the bid side.

### Efficient Computation

Decompose $G$ into a permanent accumulator ($G(\infty) = 1$) and $k$ exponential states (weights $\frac{\alpha_i/\beta_i}{1-\|\varphi\|_1}$, rates $\beta_i$). At each event:

1. Decay states by $e^{-\beta_i \Delta t}$
2. Compute contribution: $\kappa(\bar{q}^a) - \kappa(q^a)$ for market orders, $\kappa(\bar{q}^a)$ for metaorders
3. Add contribution to permanent accumulator and to each exponential state
4. Impact = permanent + sum of exponential states

Complexity: **O(k) per event**, no root-finding needed.

## Usage

```rust
use simulation_project::conditional_impact::AggressiveImpactPath;
use simulation_project::models::MultiExponentialHawkes;

let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);

let impact = AggressiveImpactPath::from_queue_samples(
    &q_samples, &q_bar_samples, &eval_times,
    &is_market_order, &hawkes, c_kappa, d_kappa,
);
```
