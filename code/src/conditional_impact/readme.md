# Conditional Impact

Closed-form market impact computation for Hawkes process-driven limit order books. This module implements two complementary pricing models for market impact: the **Flow Imbalance Model** (passive impact from queue expectations) and the **Hybrid Propagator Model** (aggressive impact from propagated metaorders plus instantaneous queue corrections).

## Overview

Both models assume:
- Intensity functions $\lambda^L$ (decreasing in $q$) and $\lambda^C$ (increasing in $q$) are affine in queue state $q$.
- The net intensity difference is $\lambda^L - \lambda^C = d_\lambda - c_{\lambda} \cdot q$ with decay rate $c_{\lambda} > 0$.
- The passive (flow imbalance) model additionally requires $\kappa(q)$ to be affine: $\kappa(q) = c_\kappa q + d_\kappa$, enabling a closed-form impact formula.
- The aggressive hybrid model accepts a queue-correction function $\kappa$ and a constant propagated metaorder weight $\bar{\kappa}$.

The module provides closed-form solutions and O(k) efficient computation for multi-exponential Hawkes kernels.

## Directory Structure

```
conditional_impact/
├── impact_utils/
│   ├── propagator.rs      # Hawkes kernel inversion
│   ├── tail_intensity.rs  # Integrated conditional intensity
│   └── impact_factors.rs  # Tail impact at event times
├── flow_imbalance_model/  # Passive impact via queue expectation
│   ├── single_queue/
│   │   └── impact_path.rs     # Single-queue impact trajectory
│   ├── multi_queue/
│   │   └── bidask_impact.rs   # Bid-ask impact computation
│   └── README.md              # Detailed flow imbalance model documentation
├── propagator_model/      # Hybrid aggressive impact via propagator price kernel
│   ├── mod.rs             # AggressiveImpactPath implementation
│   └── README.md              # Detailed propagator model documentation
└── README.md (this file)  # Module overview and navigation
```

## Model Overview

### Flow Imbalance Model
**Location**: `flow_imbalance_model/`
**Purpose**: Compute passive market impact from queue expectation dynamics.
**Key Insight**: The underlying price is determined by the conditional expectation of queue differences, which decay according to affine mean-reverting dynamics.

**Main Classes**:
- `TailImpact` — Pre-computed tail intensity coefficients for efficient evaluation
- `ImpactPath` — Single-queue impact trajectory
- `BidAskTailImpact`, `BidAskImpactPath` — Bid-ask extension with coupled queue dynamics

**See**: [flow_imbalance_model/README.md](flow_imbalance_model/README.md) for detailed formulas and theory.

### Hybrid Propagator Model
**Location**: `propagator_model/`
**Purpose**: Compute aggressive market impact via propagated metaorders and instantaneous queue corrections.
**Key Insight**: Metaorder events respond through a propagator kernel $G(t) = 1 + \sum_i \frac{\alpha_i/\beta_i}{1-\|\varphi\|_1} e^{-\beta_i t}$ with constant weight $\bar{\kappa}$, while ordinary market-order corrections accumulate as $\kappa(\bar{q})-\kappa(q)$ without temporal decay.

**Main Classes**:
- `AggressiveImpactPath` — Impact trajectory for aggressive orders that reduce queue

**See**: [propagator_model/README.md](propagator_model/README.md) for detailed formulas and theory.

## Quick Start

### Flow Imbalance Model (Single Queue)

```rust
use simulation_project::conditional_impact::{TailImpact, ImpactPath};

// Compute tail impact at each market order time
let tail_impact = TailImpact::from_affine_queue(
    mu, alpha, beta, b_l, b_c, market_order_times
);

// Full impact trajectory
let impact = ImpactPath::new(&q_path, &bar_q_path, &tail_impact);

for (i, val) in impact.impact_path.iter().enumerate() {
    println!("Event {}: I = {:.4}", i, val);
}
```

### Hybrid Propagator Model (Aggressive Impact)

```rust
use simulation_project::conditional_impact::AggressiveImpactPath;
use simulation_project::models::MultiExponentialHawkes;

// Set up Hawkes model (propagator derived internally from alpha, beta)
let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);

// Compute aggressive impact for given queue samples
let c_kappa = 0.001_f64;
let kappa = |q: f64| -c_kappa * q;
let bar_kappa = 0.01_f64;

let impact_path = AggressiveImpactPath::from_queue_samples(
    &q_samples,
    &q_bar_samples,
    &eval_times,
    &is_market_order,
    &hawkes,
    &kappa,
    bar_kappa,
);

for (i, val) in impact_path.impact_path.iter().enumerate() {
    println!("Time {}: MI = {:.4}", i, val);
}
```

## Detailed Documentation

- **[Flow Imbalance Model](flow_imbalance_model/README.md)** — Passive impact via queue expectation dynamics
  - Single-queue impact formula
  - Tail intensity computation
  - Bid-ask extension

- **[Hybrid Propagator Model](propagator_model/README.md)** — Aggressive impact via propagated metaorders and queue corrections
  - Martingale propagator derivation
  - Impact computation

## Performance

| Operation | Complexity |
|-----------|------------|
| Resolvent roots (passive model) | O(k²) where k = # exponential components |
| TailImpact precompute | O(nk) where n = # events |
| ImpactPath evaluation | O(n) |

Precomputation enables linear-time impact evaluation on top of event sequences.
