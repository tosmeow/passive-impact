# Simulation

Ogata's thinning algorithm for point processes with Markovian intensity.

## Structure

```
simulation/
├── simulator.rs                      # Core thinning algorithm
├── conditional_simulator.rs          # Conditional path sampling
└── conditional_simulator_extensions/
    ├── efficient.rs                  # Efficient conditional simulation
    └── multiple.rs                   # Multiple path simulation
```

## Thinning Algorithm

For intensity $\lambda_t$ with upper bound $\bar\lambda$:

1. Sample inter-arrival from $\text{Exp}(\bar\lambda)$
2. Accept with probability $\lambda_t / \bar\lambda$
3. Update Markovian state on acceptance
4. For multivariate: select dimension proportional to $(\lambda^1_t, \ldots, \lambda^d_t)$

```rust
use simulation_project::simulation::{simulate, simulate_with_externals};

// Basic simulation
let result = simulate(&process, t_max, Some(seed));

// With external events (deterministic, update state but bypass thinning)
let result = simulate_with_externals(&process, t_max, &external_events, Some(seed));
```

## Conditional Simulation

Sample counterfactual paths conditioned on observed baseline.

```rust
use simulation_project::simulation::ConditionalSimulationContext;

let ctx = ConditionalSimulationContext::new(
    &process,
    &cond_events_by_dim,      // Events to condition on
    cond_external.as_ref(),   // External events in conditioning path
    new_external.as_ref(),    // External events in new path
    time_horizon,
);

let result = ctx.simulate(Some(seed));
```

**Coupling**: Market orders (dim 2) are shared; limits/cancels (dims 0,1) are sampled independently.

## Efficient Simulation

Impact computation requires queue states $(q_t, \bar{q}_t)$ only at market order times $\{T_n\}$.

**Optimization**: Record state values exclusively at these event times, avoiding full path storage.

- Storage: fixed-size vectors indexed by market order sequence (vs. variable-length paths)
- Logic: streamlined conditional simulation targeting specific observation points

## Parallelization

Individual simulations are sequential; batch runs parallelize via rayon.

See [`simulation_helpers`](../simulation_helpers/) for:
- `ParallelSimulator`: Single queue batch simulation
- `BidAskParallelSimulator`: Bid-ask queue batch simulation

```rust
use simulation_project::simulation_helpers::ParallelSimulator;

let simulator = ParallelSimulator {
    process: &process,
    cond_events_by_dim: &events_by_dim,
    cond_external_events: cond_external.as_ref(),
    new_external_events: new_external.as_ref(),
    time_horizon,
    initial_queue_size,
    reference_path: &q_path,
    tail_impact: &tail_impact,
    market_orders: &market_orders,
    simulating_bar_q: true,
};

let results = simulator.run(n_simulations);
```
