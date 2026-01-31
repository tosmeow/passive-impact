# Conditional Impact Module

Closed-form computation of expected market impact using Hawkes propagator theory.

## Overview

This module computes the **conditional expected impact** of trading activity on a limit order book queue. The key result is an analytical formula for:

$$I(t) = \int_0^t (\bar{q}_s - q_s) \, dN_s + (\bar{q}_t - q_t) \cdot \underbrace{\int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds}_{\text{tail impact}}$$

where $q$ and $\bar{q}$ are baseline and counterfactual queue paths, and $N$ is the market order Hawkes process.

## Architecture

```
conditional_impact/
├── propagator.rs      # Hawkes kernel inversion
├── tail_intensity.rs  # Integrated conditional intensity
├── impact_factors.rs  # Tail impact at event times
└── impact_path.rs     # Full impact trajectory
```

## Mathematical Framework

### The Propagator

For a Hawkes process with kernel $\varphi$, the **propagator** (or resolvent) $\psi$ satisfies:

$$\psi = \varphi + \varphi * \psi$$

Equivalently, $\psi = (\delta_0 - \varphi)^{-1} - \delta_0$ in operator notation.

For a **multi-exponential kernel**:

$$\varphi(s) = \sum_{i=1}^{k} \alpha_i e^{-\beta_i s}$$

The propagator is also a sum of exponentials:

$$\psi(s) = \sum_{j=1}^{k} c_j e^{-\lambda_j s}$$

where $\lambda_j$ are the roots of the characteristic equation:

$$1 - \sum_{i=1}^{k} \frac{\alpha_i}{x + \beta_i} = 0$$

and $c_j$ are computed via finite differences.

### Tail Intensity

The **tail intensity** with exponential decay $c_\lambda$ is:

$$\mathcal{I}_t = \int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds$$

Using the propagator, this admits the closed form:

$$\mathcal{I}_t = \sum_{i=1}^{k} F_i \cdot R^i_t + \frac{\mu}{c_\lambda} + \mu \sum_{j=1}^{k} \frac{c_j}{\lambda_j} \left( \frac{1}{c_\lambda} - \frac{1}{\lambda_j + c_\lambda} \right)$$

where:
- $R^i_t$ are the Markovian state variables of the Hawkes process
- $F_i$ are precomputed **tail intensity factors**:

$$F_i = \sum_{j=1}^{k} \frac{c_j}{\lambda_j - \beta_i} \left( \frac{1}{c_\lambda + \beta_i} - \frac{1}{c_\lambda + \lambda_j} \right) + \frac{1}{c_\lambda + \beta_i}$$

### Impact Path

The full impact at time $t$ combines:

1. **Realized impact:** $\sum_{s \leq t} (\bar{q}_s - q_s)$ — cumulative queue displacement at past market orders
2. **Expected future impact:** $(\bar{q}_t - q_t) \cdot \mathcal{I}_t$ — current displacement times tail intensity

## Module Components

### `Propagator`

Computes the propagator coefficients $(\lambda_j, c_j)$ from Hawkes parameters:

```rust
pub struct Propagator {
    pub hawkes_params: MultiExponentialHawkes,
    pub lambda: Vec<f64>,  // Propagator decay rates
    pub c: Vec<f64>,       // Propagator amplitudes
}

impl Propagator {
    pub fn new(hawkes_params: MultiExponentialHawkes) -> Self;
}
```

**Root-finding:** Uses the IVT solver to find roots of $1 - \sum \frac{\alpha_i}{x + \beta_i} = 0$ in each interval $(-\beta_{i+1}, -\beta_i)$.

### `TailIntensity`

Precomputes factors for efficient tail intensity evaluation:

```rust
pub struct TailIntensity {
    pub hawkes_params: MultiExponentialHawkes,
    pub c_lambda: f64,      // Impact decay rate
    pub factors: Vec<f64>,  // Precomputed Fᵢ
    pub lambda: Vec<f64>,   // From propagator
    pub c: Vec<f64>,        // From propagator
}

impl TailIntensity {
    pub fn new(hawkes_params: MultiExponentialHawkes, c_lambda: f64) -> Self;

    /// Compute tail intensity given current Markovian state
    pub fn compute(&self, state: &[f64]) -> f64;
}
```

**Usage:**
```rust
let tail = TailIntensity::new(hawkes, c_lambda);
let intensity = tail.compute(&current_state);  // O(k) computation
```

### `TailImpact`

Evaluates tail intensity at each event time in a Hawkes path:

```rust
pub struct TailImpact {
    pub events: Vec<f64>,           // Event timestamps
    pub tail_impact_events: Vec<f64>, // Iₜ at each event
}

impl TailImpact {
    pub fn new(
        events: &[f64],
        hawkes_params: &MultiExponentialHawkes,
        c_lambda: f64,
    ) -> Self;
}
```

### `ImpactPath`

Combines all components to compute the full impact trajectory:

```rust
pub struct ImpactPath {
    pub impact_path: Vec<f64>,  // I(t) at each event time
}

impl ImpactPath {
    pub fn new(
        queue_path: &QueuePath,      // Baseline q
        queue_bar_path: &QueuePath,  // Counterfactual q̄
        tail_impact: &TailImpact,
    ) -> Self;
}
```

## Usage Example

```rust
use models::{MultiExponentialHawkes, AffineQueueProcess, QueueProcess};
use conditional_impact::{TailImpact, ImpactPath};

// 1. Define Hawkes parameters
let hawkes = MultiExponentialHawkes::new(
    0.5,
    vec![0.065, 0.2, 0.325, 0.65],
    vec![0.15, 0.60, 2.5, 10.0],
);

// 2. Compute c_lambda from queue parameters
let c_lambda = AffineQueueProcess::c_lambda(-0.275, 0.125);  // b_c - b_l

// 3. Extract market order times from simulation
let market_order_times: Vec<f64> = result.events_per_dim[2].clone();

// 4. Compute tail impact at each market order
let tail_impact = TailImpact::new(&market_order_times, &hawkes, c_lambda);

// 5. Compute full impact path
let impact = ImpactPath::new(&baseline_queue, &counterfactual_queue, &tail_impact);

// 6. Analyze impact trajectory
for (i, impact_value) in impact.impact_path.iter().enumerate() {
    println!("Event {}: I(t) = {:.4}", i, impact_value);
}
```

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `Propagator::new` | $O(k^2)$ | $k$ root-finding operations |
| `TailIntensity::new` | $O(k^2)$ | Precompute $k$ factors |
| `TailIntensity::compute` | $O(k)$ | Per-event evaluation |
| `TailImpact::new` | $O(nk)$ | $n$ events, $k$ components |
| `ImpactPath::new` | $O(n)$ | Linear in events |

The precomputation in `Propagator` and `TailIntensity` enables **constant-time** tail intensity evaluation at each event, making the overall algorithm linear in the number of events.

## The $c_\lambda$ Parameter

The decay rate $c_\lambda = b_c - b_l$ arises from the affine queue dynamics:

- $b_l < 0$: Limit order intensity decreases with queue size (crowding out)
- $b_c > 0$: Cancellation intensity increases with queue size

Thus $c_\lambda > 0$ ensures impact decays over time as the queue mean-reverts.

**Physical interpretation:** A queue displacement $\delta q$ has expected future impact proportional to $\delta q \cdot e^{-c_\lambda \tau}$ at horizon $\tau$.
