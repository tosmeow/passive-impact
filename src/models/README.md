# Models Module

Core abstractions for multivariate point processes with Markovian intensity dynamics.

## Overview

This module provides the building blocks for defining and simulating intensity-based point processes. The design follows a **trait-based abstraction** that separates the mathematical specification from simulation mechanics.

## Architecture

```
models/
├── multivariate_process.rs   # Core trait and data structures
├── markovian_process.rs      # Generic Markovian implementation
├── hawkes_processes.rs       # Multi-exponential Hawkes
└── queue_processes.rs        # Order book queue dynamics
```

## Core Trait

### `MultivariateMarkovianIntensity`

The fundamental abstraction for any $d$-dimensional point process with $k$-dimensional Markovian state:

```rust
pub trait MultivariateMarkovianIntensity {
    fn lambda(&self) -> Vec<f64>;           // Current intensity vector λ = (λ¹, ..., λᵈ)
    fn lambda_bar(&self) -> f64;            // Upper bound: λ̄ ≥ Σᵢ λⁱ
    fn update_state(&mut self, event: &MultivariateEvent);
    fn get_state(&self) -> Vec<f64>;        // Markovian factors (R¹, ..., Rᵏ)
    fn dim(&self) -> usize;                 // Process dimension d
    fn reset(&mut self);
}
```

## Data Structures

### `MultivariateEvent`

A single event in a multivariate point process:

```rust
pub struct MultivariateEvent {
    pub time: f64,    // Event timestamp
    pub dim: usize,   // Dimension (0-indexed)
}
```

### `MultivariateSimulationResult`

Container for simulation output, organized by dimension:

```rust
pub struct MultivariateSimulationResult {
    pub events: Vec<MultivariateEvent>,      // All events (time-ordered)
    pub events_per_dim: Vec<Vec<f64>>,       // Events grouped by dimension
}
```

## Hawkes Processes

### `MultiExponentialHawkes`

A univariate Hawkes process with multi-exponential kernel:

$$\lambda_t = \mu + \sum_{i=1}^{k} R^i_t, \quad R^i_t = \sum_{s < t} \alpha_i e^{-\beta_i(t-s)}$$

**Parameters:**
- `mu` — Baseline intensity
- `alpha` — Excitation amplitudes $(\alpha_1, \ldots, \alpha_k)$
- `beta` — Decay rates $(\beta_1, \ldots, \beta_k)$

**Stationarity condition:** $\sum_i \frac{\alpha_i}{\beta_i} < 1$

**State updates:** On each event at time $t$, the Markovian factors update as:

$$R^i_{t^+} = R^i_{t^-} \cdot e^{-\beta_i \Delta t} + \alpha_i$$

where $\Delta t$ is the time since the last event.

```rust
let hawkes = MultiExponentialHawkes::new(
    0.5,                              // μ
    vec![0.1, 0.2, 0.3],              // α
    vec![1.0, 2.0, 5.0],              // β
);
```

## Queue Processes

### `QueueEvent` & `QueuePath`

Representation of limit order book queue evolution:

```rust
pub struct QueueEvent {
    pub queue_event: u32,   // Event type: 0=limit, 1=cancel, 2=market
    pub queue_size: u32,    // Queue size after event
    pub time: f64,
}

pub struct QueuePath {
    pub events: Vec<QueueEvent>,
}
```

### `QueueProcess`

Generic queue with three event types:
- **Dimension 0:** Limit orders (queue increases)
- **Dimension 1:** Cancellations (queue decreases)
- **Dimension 2:** Market orders (queue decreases)

### `AffineQueueProcess`

Specialized queue where limit and cancel intensities are **affine in queue size**:

$$\lambda^L(q) = a_l + b_l \cdot q, \quad \lambda^C(q) = a_c + b_c \cdot q$$

Market orders follow an independent Hawkes process.

```rust
let process = AffineQueueProcess::new(
    100,    // q₀: initial queue
    50.0,   // aₗ: limit baseline
    -0.1,   // bₗ: limit sensitivity (negative = crowding)
    5.0,    // aᶜ: cancel baseline
    0.05,   // bᶜ: cancel sensitivity
    0.5,    // μ: Hawkes baseline
    alpha,  // Hawkes α
    beta,   // Hawkes β
);
```

**Important constant:** $c_\lambda = b_c - b_l$ governs the decay rate of conditional impact.

## Generic Markovian Process

### `MarkovianProcess`

A flexible wrapper that accepts closures for intensity and state dynamics:

```rust
pub struct MarkovianProcess {
    lambda: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    lambda_bar: Box<dyn Fn(&[f64]) -> f64>,
    state_update: Box<dyn Fn(&mut Vec<f64>, &MultivariateEvent)>,
    // ...
}
```

This enables defining custom processes without implementing the full trait:

```rust
let process = MarkovianProcess::new(
    |state| vec![state[0], state[1]],           // λ(state)
    |state| state.iter().sum::<f64>() * 1.5,    // λ̄(state)
    |state, event| { /* update logic */ },       // state transition
    || vec![0.0, 0.0],                          // initial state
    2,                                          // dimension
);
```

## Usage Examples

### Simulating and Extracting Events

```rust
use models::{MultiExponentialHawkes, MarkovianProcessSimulator};

let hawkes = MultiExponentialHawkes::new(0.5, vec![0.2], vec![1.0]);
let result = hawkes.simulate(100.0, Some(42));

// Access all events
for event in &result.events {
    println!("t={:.4}, dim={}", event.time, event.dim);
}

// Access by dimension
let market_order_times = &result.events_per_dim[0];
```

### Converting Simulation to Queue Path

```rust
use models::{QueueProcess, AffineQueueProcess};

let process = AffineQueueProcess::new(/* ... */);
let result = process.simulate(250.0, None);

let queue_path = QueueProcess::result_to_queue_path(&result, 100);

for event in &queue_path.events {
    println!("t={:.2}: queue={}", event.time, event.queue_size);
}
```
