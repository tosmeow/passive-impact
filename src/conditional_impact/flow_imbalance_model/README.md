# Flow Imbalance Model

Passive market impact computation based on queue expectation dynamics.

## Overview

In this model, the underlying price is determined by the conditional expectation of queue differences:

$$P_t = \mathbb{E}_t\left[\int_0^\infty \kappa(q^a_s)\, dN^a_s - \kappa(q^b_s) \, dN^b_s\right]$$

where:
- $\bar{q}$ is the counterfactual queue (e.g., with a limit order posted)
- $q$ is the baseline queue
- $\kappa(x) = c_\kappa x + d_\kappa$ is an affine impact function
- $dN$ represents market order arrivals

The queue difference $\bar{q}_s - q_s$ decays mean-revertingly due to affine intensity dynamics, enabling closed-form computation via **tail intensity**.

## Single-Queue Impact Formula

### Impact Decomposition

The general impact formula is:

$$I_t = c_\kappa \mathbb{E}_t\left[\int_0^\infty (\bar{q}_s - q_s) \, dN_s\right]$$

We decompose this into realized and future components:

$$I(t) = c_\kappa \int_0^t (\bar{q}_s - q_s) \, dN_s + c_\kappa(\bar{q}_t - q_t) \cdot \mathcal{I}_t$$

where:
- **Realized part**: $\int_0^t (\bar{q}_s - q_s) \, dN_s$ is the historical contribution
- **Tail part**: $\mathcal{I}_t$ is the future contribution weighted by tail intensity

### Tail Intensity

The tail intensity captures the expected future impact discounted by the decay of the queue difference:

$$\mathcal{I}_t = \int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds$$

where $c_\lambda = b_c - b_l > 0$ is the decay rate:
- $b_l < 0$: crowding reduces limit order arrival
- $b_c > 0$: larger queues see more cancellations

### Closed-Form Computation

For a multi-exponential Hawkes kernel $\phi(t) = \mu + \sum_i \alpha_i e^{-\beta_i t}$, we pre-compute tail intensity as:

$$\mathcal{I}_t = C + \sum_{i=1}^{k} F_i \cdot R^i_t$$

where $R^i_t = \sum_{s < t} \alpha_i e^{-\beta_i(t-s)}$ are the Markovian state variables of the Hawkes process.

The constants are:

$$C = \frac{\mu}{c_\lambda} + \mu \sum_{j=1}^{k} \frac{c_j}{\lambda_j} \left( \frac{1}{c_\lambda} - \frac{1}{\lambda_j + c_\lambda} \right)$$

$$F_i = \sum_{j=1}^{k} \frac{c_j}{\lambda_j - \beta_i} \left( \frac{1}{c_\lambda + \beta_i} - \frac{1}{c_\lambda + \lambda_j} \right) + \frac{1}{c_\lambda + \beta_i}$$

where $\lambda_j$ are the roots of the characteristic equation (see [Propagator](../propagator_model/README.md)).

This decomposition enables **O(k) evaluation** at each event by updating the Markovian states incrementally.

## Bid-Ask Extension

### Coupled Queue Dynamics

With both bid and ask queues, the intensity functions become:

$$\lambda^L(q^a, q^b) = a_L q^a + b_L q^b + d_L$$
$$\lambda^C(q^a, q^b) = a_C q^a + b_C q^b + d_C$$

The price dynamics reflect both queue impacts:

$$P_t = \mathbb{E}_t\left[\int_0^\infty \kappa(q^a_s) \, dN^a_s - \kappa(q^b_s) \, dN^b_s\right]$$

### Impact Formula

When posting a limit order at the bid, we impact both queues:

$$I_t = c_\kappa \mathbb{E}_t\left[\int_0^\infty (\bar{q}^a_s - q^a_s) \, dN^a_s - (\bar{q}^b_s - q^b_s) \, dN^b_s\right]$$

We decompose for each queue independently:

$$I^x(t) = c_\kappa \int_0^t (\bar{q}^x_s - q^x_s) \, dN^x_s + c_\kappa \mathcal{I}^x_t(q_t, \bar{q}_t)$$

Then take the difference: $I = I^a - I^b$.

### Eigendirection Analysis

The queue difference matrix evolves as:

$$\mathbb{E}_t[\bar{q}_s - q_s] = (\bar{q}_t - q_t) - \int_t^s C_{\lambda} \mathbb{E}_t[\bar{q}_u - q_u] \, du$$

where the coupling matrix is:

$$C_{\lambda} = \begin{pmatrix} a_C - a_L & b_C - b_L \\ b_C - b_L & a_C - a_L \end{pmatrix}$$

This symmetric matrix has two eigendirections with eigenvalues:

- $u_0 = (1, 1)$ with eigenvalue $c_0 = (a_C - a_L) + (b_C - b_L)$
- $u_1 = (1, -1)$ with eigenvalue $c_1 = (a_C - a_L) - (b_C - b_L)$

The queue difference decomposes as:

$$\mathbb{E}_t[\bar{q}_s - q_s] = \alpha_0(t) u_0 e^{-c_0(s-t)} + \alpha_1(t) u_1 e^{-c_1(s-t)}$$

where $\alpha_0$ and $\alpha_1$ are computed from the initial queue difference via projection.

Each eigendirection contributes independently to the tail impact, as in the single-queue case but with two separate decay rates.

## Implementation Details

### Classes

- **`TailImpact`** — Single-queue tail intensity pre-computation
  - Stores $C$ and $F_i$ coefficients
  - Updates Markovian states $R^i$ at each event

- **`ImpactPath`** — Single-queue impact trajectory
  - Combines realized and tail contributions
  - O(n) evaluation

- **`SymmetricCMatrix`** — Bid-ask coupling matrix
  - Computes and stores eigenvalues $c_0$, $c_1$

- **`BidAskTailImpact`** — Bid-ask tail intensity per queue and eigendirection

- **`BidAskImpactPath`** — Bid-ask impact trajectory

### Usage Example

```rust
use simulation_project::conditional_impact::{TailImpact, ImpactPath};

// Pre-compute tail intensity (done once)
let tail_impact = TailImpact::from_affine_queue(
    mu,      // Hawkes baseline intensity
    alpha,   // Exponential decay amplitudes
    beta,    // Exponential decay rates
    b_l,     // Limit order sensitivity to queue
    b_c,     // Cancellation sensitivity to queue
    market_order_times
);

// Compute impact path for given queue trajectory
let impact = ImpactPath::new(&q_path, &bar_q_path, &tail_impact);

// Access results
for (i, val) in impact.impact_path.iter().enumerate() {
    println!("Event {}: Impact = {:.4}", i, val);
}
```

### Bid-Ask Example

```rust
use simulation_project::conditional_impact::{
    BidAskTailImpact, BidAskImpactPath, SymmetricCMatrix
};

// Set up coupling matrix for symmetric Hawkes
let c_matrix = SymmetricCMatrix::from_affine_symmetric(
    b_l_own,   // Own-queue limit order effect
    b_l_cross, // Cross-queue limit order effect
    b_c_own,   // Own-queue cancellation effect
    b_c_cross  // Cross-queue cancellation effect
);

// Pre-compute tail intensity
let tail_impact = BidAskTailImpact::new_symmetric_hawkes(
    mu, alpha, beta, c_matrix, events_a, events_b
);

// Compute bid-ask impact
let impact = BidAskImpactPath::new(&q_a, &q_b, &bar_q_a, &bar_q_b, &tail_impact);

// Access bid-ask impact difference
for (i, val) in impact.impact_path.iter().enumerate() {
    println!("Event {}: Bid-Ask Impact = {:.4}", i, val);
}
```

## Complexity Analysis

| Operation | Complexity | Description |
|-----------|-----------|-------------|
| Pre-computation | O(nk) | Update $k$ Markovian states at each of $n$ events |
| Tail intensity update | O(k) | Evaluate $C + \sum_i F_i R^i$ at each event |
| Impact path | O(n) | Linear evaluation given pre-computed tail intensity |

The pre-computation phase dominates for large event counts, but subsequent evaluations (e.g., for different queue paths) are linear in $n$.

## Notes

- All implementation uses $c_\kappa = 1$ internally; scale results as needed
- Queue differences are required; raw queues alone are insufficient
- Tail intensity coefficients depend only on Hawkes parameters, not on queue trajectory
- Bid-ask implementation assumes symmetric Hawkes structure (same parameters for both queues)
