# Models

Core abstractions for multivariate point processes with Markovian intensity.

The technical assumption on the processes for correctness of the core simulation is:

- For a process d-dimensional $X$, the intensities vector path $(\lambda^X_s)_{s \in [t, T]}$ conditionned on the fact that $X$ has no event in $[t,T]$ is a non-increasing function in each coordinate.

## Structure

```
models/
├── hawkes/
│   └── hawkes_processes.rs       # Multi-exponential Hawkes
├── processes/
│   ├── multivariate_process.rs   # Core trait and data structures
│   └── markovian_process.rs      # Generic closure-based implementation
└── queues/
    ├── queue_processes.rs        # Affine queue dynamics (single queue)
    └── multiqueue_processes.rs   # Bid-ask queue dynamics (double queue)
```

## Core Trait

```rust
pub trait MultivariateMarkovianIntensity {
    fn lambda(&self) -> Vec<f64>;      // Current intensities (λ¹, ..., λᵈ)
    fn lambda_bar(&self) -> f64;       // Upper bound for thinning
    fn update_state(&mut self, event: &MultivariateEvent);
    fn get_state(&self) -> Vec<f64>;   // Markovian factors
    fn dim(&self) -> usize;
    fn reset(&mut self);
}
```

## Data Structures

```rust
pub struct MultivariateEvent {
    pub time: f64,
    pub dim: usize,
}

pub struct MultivariateSimulationResult {
    pub events: Vec<MultivariateEvent>,
    pub events_per_dim: Vec<Vec<f64>>,
}
```

## Hawkes Process

Multi-exponential kernel with Markovian state: $\lambda_t = \mu + \sum_{i=1}^{k} R^i_t$, with $R^i_t = R^i_0 + \int_0^{t-} \alpha_i e^{-\beta_i (t-s)} dN_s$.

We allow to start the Hawkes simulation from a point where the latent Markovian factors $(R^i)_{i=1}^k$ are at the stationary value, this avoids warming up the Hawkes process for the queue simulation on a burning period: warming up the process boils down to having started at $R^i_0 = 0$ until the values become closer to stationary.

```rust
let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);

// With stationary initial state
let hawkes = MultiExponentialHawkes::new_with_state(
    MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
    mu, alpha, beta,
);
```

Stationarity: $\sum_i \alpha_i / \beta_i < 1$

## Queue Process

Three dimensions: limits (0), cancels (1), markets (2).

**Affine intensities**:
$$\lambda^L(q) = a_l + b_l \cdot q, \quad \lambda^C(q) = a_c + b_c \cdot q$$

The general method for queue generation with ::new simulates the Hawkes process as well.

We provide a method ::new_queue that will set the effective market order intensity to zero. It is used when we want to specify a Hawkes path trajectory that will be unchanged in conditional simulations and provide more efficient code.

```rust
// Coupled: queue + Hawkes state
let process = AffineQueueProcess::new(q0, a_l, b_l, a_c, b_c, mu, alpha, beta);

// Decoupled: queue state only (Hawkes injected externally)
let process = AffineQueueProcess::new_queue(q0, a_l, b_l, a_c, b_c);
```

Key constant: $c_\lambda = b_c - b_l$ governs impact decay.

```rust
let c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
```

## Bid-Ask Queue Process

For symmetric bid-ask dynamics with coupled queues:

```rust
let process = BidAskQueueProcess::new(
    q0_bid, q0_ask,
    a_l, b_l, a_c, b_c,
    mu, alpha, beta,
);
```

Five dimensions: bid limits (0), bid cancels (1), ask limits (2), ask cancels (3), markets (4).

## Queue Path

```rust
pub struct QueueEvent {
    pub queue_event: u32,  // 0=limit, 1=cancel, 2=market
    pub queue_size: u32,
    pub time: f64,
}

pub struct QueuePath {
    pub events: Vec<QueueEvent>,
}

// Convert simulation result to queue path
let path = AffineQueueProcess::result_to_queue_path(&result, initial_size);
let q_at_t = path.queue_at_time(t);
```
