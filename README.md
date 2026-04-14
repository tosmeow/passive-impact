# Passive Market Impact Simulation

A high-performance Rust library for simulating and analyzing market impact using point processes. Combines Hawkes processes (for market orders) with queue-reactive Markovian dynamics (for limit orders and cancellations) to compute the price effect of trading strategies through conditional path simulation.

## Visual Overview

<p align="center">
  <img src="python/experiments/single_queue_impact/images/impact_given_q.png" width="48%" alt="Conditional impact distribution given baseline queue"/>
  <img src="python/experiments/single_queue_impact/images/queue_given_q.png" width="48%" alt="Conditional queue distribution given baseline"/>
</p>
<p align="center">
  <img src="python/experiments/single_queue_impact/images/impact_given_qbar.png" width="48%" alt="Impact given shocked queue"/>
  <img src="python/experiments/single_queue_impact/images/queue_given_qbar.png" width="48%" alt="Queue given shocked queue"/>
</p>

*Conditional simulation of 500 counterfactual market paths (gray shading) with empirical mean (red) and observed baseline (black). Each panel shows a different initial queue state.*

## What This Library Provides

- **Exact conditional simulation** of coupled point processes, enabling pathwise comparison of observed vs. counterfactual market scenarios.
- **Queue-reactive order dynamics** with affine intensity functions, modeling how limit orders, cancellations, and market orders respond to queue depth.
- **Closed-form market impact computation** for Hawkes with kernels as sum of exponentials using resolvant operator methods, enabling efficient impact estimation without nested Monte Carlo.
- **Flexible architecture** supporting both single-queue and bid-ask queue pair scenarios, with optimized ("efficient") and general simulation variants.

## Quick Start

### Hawkes Process

```rust
use simulation_project::models::MultiExponentialHawkes;
use simulation_project::simulation::simulate;

let mu = 1.0;
let alpha = vec![0.065, 0.2, 0.325, 0.65];
let beta = vec![0.15, 0.60, 2.5, 10.0];

let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);
let result = simulate(&hawkes, 100.0, Some(42));
```

### Affine Queue

```rust
use simulation_project::models::AffineQueueProcess;
use simulation_project::simulation::simulate_with_externals;

// λ^L(q) = a_l + b_l·q,  λ^C(q) = a_c + b_c·q
// Decoupled mode: market orders are injected as external events (hawkes_result).
let process = AffineQueueProcess::new_queue(
    200.0,   // initial queue
    100.0, -0.275,  // a_l, b_l
    2.0, 0.125,     // a_c, b_c
);

let result = simulate_with_externals(&process, 100.0, &hawkes_as_market, None);
let queue_path = AffineQueueProcess::result_to_queue_path(&result, 200);
```

### Conditional Impact

```rust
use simulation_project::conditional_impact::{TailImpact, ImpactPath};

let tail_impact = TailImpact::from_affine_queue(
    mu, alpha, beta, b_l, b_c, market_order_times
);
// ImpactPath::new takes queue paths by value
let impact = ImpactPath::new(q_path, bar_q_path, &tail_impact);
```

## Mathematical Background

### Hawkes Process

Conditional intensity with multi-exponential kernel:

$$\lambda_t = \mu + \int_0^{t-} \phi(t-s)dN_s = \mu + \sum_{i=1}^{k} R^i_t, \quad \varphi(s) = \sum_{i=1}^{k} \alpha_i e^{-\beta_i s}$$

with Markovian states $R^i_t := \int_0^{t-} \alpha_i e^{-\beta_i(t-s)} dN_s$ enabling O(1) intensity updates.

### Queue Dynamics

- **Limits**: $\lambda^L(q) = a_l + b_l \cdot q$, with $b_l < 0$.
- **Cancels**: $\lambda^C(q) = a_c + b_c \cdot q$ with $b_c > 0$.
- **Markets**: Hawkes process.

### Conditional Impact

With $c_\lambda := b_c - b_l$, we implement:

$$I(t) = c_\kappa \int_0^t (\bar{q}_s - q_s) \, dN_s + c_\kappa (\bar{q}_t - q_t) \cdot \mathcal{I}_t$$

where the following term admits a closed form relying on the resolvent operator $(\delta_0 - \varphi)^{-1}$:
```math
\mathcal{I}_t = \int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds
```

## Modules

| Module | Description |
|--------|-------------|
| [`models`](src/models/) | Hawkes, queues, Markovian abstractions |
| [`simulation`](src/simulation/) | Thinning algorithm, conditional simulation |
| [`simulation_helpers`](src/simulation_helpers/) | Parallel batch simulation, event utilities |
| [`conditional_impact`](src/conditional_impact/) | Resolvent and propagator impact models |
| [`utils`](src/utils/) | IVT root-finding, finite differences |

## Experiments

All experiments follow the same pattern: (1) run a Rust binary to generate simulation data, (2) use the corresponding Python notebook to analyze and visualize. Simulation data is saved as `.npy` files in the `data/` directory.

| Experiment | What it shows | Commands |
|---|---|---|
| **Single Queue Impact** | Passive market impact from limit-order metaorders in a single-sided queue | `cargo run --release --bin single_queue_efficient_{with,without}_us` <br/> [results](python/experiments/single_queue_impact/) |
| **Double Queue Impact** | Passive impact in a bid-ask queue pair; shows how impact differs across sides | `cargo run --release --bin double_queue_efficient_{with,without}_us` <br/> [results](python/experiments/double_queue_impact/) |
| **Aggressive Impact** | Market-order impact under the propagator price model | `cargo run --release --bin agressive_impact` <br/> [results](python/experiments/agressive_impact/) |
| **Hybrid Aggressive Impact** | Market-order impact under the hybrid price model | `cargo run --release --bin agressive_impact_hybrid` <br/> [results](python/experiments/agressive_impact_hybrid/) |

The two aggressive impact experiments use different price models for modeling aggressive (market-order) impact. The **Aggressive Impact** experiment uses a pure propagator model where the per-trade impact function $\kappa(q)$ depends on queue depth. The **Hybrid Aggressive Impact** experiment combines a queue-dependent instantaneous correction ($\kappa(q) = -c_\kappa \cdot q$) with a propagator term weighted by a constant $\bar{\kappa}$.

### General vs. Efficient Simulator Variants

For `single_queue_*` and `double_queue_*` experiments, two variants are available:

- **General** (`*_general_*`): Full event history simulation, useful for detailed analysis
- **Efficient** (`*_efficient_*`): Memory-optimized variant that computes impact without storing all queue paths

For most use cases, **use the efficient variants** (`*_efficient_*`). The general variants are provided for research and validation purposes.

### Running the Experiments

#### Option 1: Run a single experiment
```bash
# Generate data for single queue impact
cargo run --release --bin single_queue_efficient_with_us
cargo run --release --bin single_queue_efficient_without_us

# Generate plots
cd python/experiments/single_queue_impact
python plot_utils.py
```

#### Option 2: Run all experiments
```bash
cargo build --release

cargo run --release --bin single_queue_efficient_with_us
cargo run --release --bin single_queue_efficient_without_us
cargo run --release --bin double_queue_efficient_with_us
cargo run --release --bin double_queue_efficient_without_us
cargo run --release --bin agressive_impact
cargo run --release --bin agressive_impact_hybrid
```

Outputs: `.npy` files saved to `data/*/` directories, organized by experiment type.

---

## Dependencies

- `numpy`, `scipy`: Numerical computing
- `numba`: JIT compilation for performance-critical Python code
- `maturin`, `pyo3`: Rust-to-Python bindings (development only)
- `pytest`: Testing framework
