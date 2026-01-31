# Passive Impact Simulation

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Rust library for the implementation of the passive market impact framework, with a markovian limit order book model and hawkes processes for market orders.

## Example Output

<p align="center">
  <img src="python/impact_given_q.png" width="48%" alt="Impact given baseline q"/>
  <img src="python/queue_given_q.png" width="48%" alt="Queue given baseline q"/>
</p>
<p align="center">
  <img src="python/impact_given_qbar.png" width="48%" alt="Impact given baseline q̄"/>
  <img src="python/queue_given_qbar.png" width="48%" alt="Queue given baseline q̄"/>
</p>

*Conditional simulation of 500 counterfactual paths (gray) with mean (red) and baseline (black). Left: price impact trajectories. Right: queue size evolution.*

## Overview

This library provides a framework for conditional simulation of multidimensional point processes who are Markovianin a finite-dimensional state:

- **Queue dynamics** follow inhomogeneous Poisson processes with intensities depending on queue state.
- **Market order arrivals** follows a Hawkes process.
- **Conditional impact** is computed analytically using propagator operators.

The implementation supports multi-exponential Hawkes kernels with closed-form tail impact formulas.

The technical hypothesis here is that the intensities are decreasing with time when no events occur to allow efficient thinning.

## Features

- **Hawkes Process Simulation with sum of exponentials kernel** — Ogata's thinning algorithm with Markovian state tracking.
- **Affine Queue Processes** — Limit order book queues with state-dependent intensities affine in the queue state.
- **Conditional Simulation** — Generate counterfactual paths conditioned on observed trajectories.
- **Analytical Tail Impact** — Closed-form computation of expected future impact.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
simulation_project = { path = "." }
```

## Quick Start

### Simulating a Hawkes Process

```rust
use simulation_project::models::{MultiExponentialHawkes, MarkovianProcessSimulator};

// Create a multi-exponential Hawkes process
// Kernel: φ(s) = Σᵢ αᵢ exp(-βᵢ s)
let mu = 0.5;  // baseline intensity
let alpha = vec![0.065, 0.2, 0.325, 0.65];
let beta = vec![0.15, 0.60, 2.5, 10.0];

let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);

// Simulate until time T
let T: f64 = 100.0;
let result = hawkes.simulate(T, Some(42));
println!("Generated {} events", result.events.len());
```

### Affine Queue Process

```rust
use simulation_project::models::AffineQueueProcess;
use simulation_project::simulation::simulate;

// Queue intensities: λᴸ(q) = aₗ + bₗ·q,  λᶜ(q) = aᶜ + bᶜ·q
let initial_size: u32 = 250;
let a_l = 100.0;
let b_l = -0.275;
let a_c = 2.0;
let b_c = 0.125;
let mu = 0.5;
let alpha = vec![0.065, 0.2, 0.325, 0.65];
let beta = vec![0.15, 0.60, 2.5, 10.0];

let process = AffineQueueProcess::new(
    initial_size as f64,
    a_l, b_l, a_c, b_c,
    mu, alpha, beta,
);

let c_lambda: f64 = AffineQueueProcess::c_lambda(b_l, b_c);

let t_max: f64 = 250.0;
let result = simulate(&process, t_max, Some(123));
let queue_path = AffineQueueProcess::result_to_queue_path(&result, initial_size);
```

### Computing Conditional Impact

```rust
use simulation_project::conditional_impact::{TailImpact, ImpactPath};

// Compute tail impact factors
let tail_impact = TailImpact::new(&hawkes_events, &hawkes_params, c_lambda);

// Generate full impact path comparing baseline vs counterfactual
let impact = ImpactPath::new(&queue_path, &queue_bar_path, &tail_impact);
```

## Mathematical Background

### Hawkes Process

A Hawkes process is a self-exciting point process with conditional intensity:

$$\lambda_t = \mu + \int_0^t \varphi(t-s) \, dN_s$$

where $\varphi$ is the excitation kernel. This library implements **multi-exponential kernels**:

$$\varphi(s) = \sum_{i=1}^{k} \alpha_i e^{-\beta_i s}$$

This form admits a **Markovian representation** with $k$ state variables $R^i_t$, enabling $O(1)$ intensity updates per event.

### Queue Dynamics

The limit order book queue evolves according to:
- **Limit orders** arrive with intensity $\lambda^L(q) = a_l + b_l \cdot q$
- **Cancellations** occur with intensity $\lambda^C(q) = a_c + b_c \cdot q$
- **Market orders** arrive as a Hawkes process

### Conditional Impact

For a trader injecting limit orders at the best ask, the **conditional impact** measures the expected queue displacement:

$$I(t) = c_{\kappa}\int_0^t (\bar{q}_s - q_s) \, dN_s + c_\kappa (\bar{q}_t - q_t) \cdot \int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds$$

where:
- $q$ is the baseline queue path (without intervention)
- $\bar{q}$ is the counterfactual path (with meta-orders)
- $c_\lambda = b_c - b_l$ governs impact decay

The tail integral admits a **closed-form solution** using the Hawkes propagator.

## Module Documentation

| Module | Description |
|--------|-------------|
| [`models`](src/models/) | Process definitions: Hawkes, queues, Markovian abstractions |
| [`simulation`](src/simulation/) | Thinning algorithm and conditional simulation |
| [`conditional_impact`](src/conditional_impact/) | Propagator computation and impact analysis |
| [`utils`](src/utils/) | Numerical solvers (IVT root-finding, finite differences) |

## Running the Examples

Two binary examples demonstrate the full workflow:

```bash
# Impact analysis WITH trader meta-orders
cargo run --release --bin paths_with_us

# Baseline comparison WITHOUT meta-orders
cargo run --release --bin paths_without_us
```

Output files (NumPy binary format for fast I/O):
- `impact_paths.npy` — Impact trajectory $I(t)$ at each event
- `queue_paths.npy` — Queue size evolution over time
- `times.npy` — Time index for the above arrays

### Generating Plots

After running the simulations, generate visualization plots:

```bash
cd python
python plot_utils.py
```

This produces four PNG files showing impact and queue trajectories.

## Testing

```bash
cargo test
```

## License

MIT License — see [LICENSE](LICENSE) for details.
