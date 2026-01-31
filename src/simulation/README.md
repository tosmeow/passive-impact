# Simulation Module

Efficient algorithms for simulating multivariate point processes and conditional paths.

## Overview

This module implements **Ogata's thinning algorithm** for simulating intensity-based point processes, extended to support:

- Multivariate processes with dimension-dependent intensities
- External event injection for intervention analysis
- Conditional simulation for counterfactual path generation

## Architecture

```
simulation/
├── simulator.rs              # Core thinning algorithm
└── conditional_simulator.rs  # Conditional path sampling
```

## Thinning Algorithm

### Theory

For a point process with intensity $\lambda_t$, the **thinning algorithm** (Ogata, 1981) generates events by:

1. **Bound the intensity:** Find $\bar{\lambda} \geq \lambda_t$ for all $t$
2. **Propose candidates:** Sample inter-arrival times from $\text{Exp}(\bar{\lambda})$
3. **Accept/reject:** Keep candidate with probability $\lambda_t / \bar{\lambda}$
4. **Update state:** Modify Markovian factors based on accepted event

For multivariate processes, step 3 also selects the event dimension proportional to $(\lambda^1_t, \ldots, \lambda^d_t)$.

### Implementation

The `MarkovianProcessSimulator` trait extends any `MultivariateMarkovianIntensity`:

```rust
pub trait MarkovianProcessSimulator: MultivariateMarkovianIntensity {
    fn simulate(&self, t_max: f64, seed: Option<u64>) -> MultivariateSimulationResult;

    fn simulate_with_externals(
        &self,
        t_max: f64,
        external_trajectory: &[(f64, usize)],
        seed: Option<u64>,
    ) -> MultivariateSimulationResult;
}
```

### Basic Simulation

```rust
use models::{MultiExponentialHawkes, MarkovianProcessSimulator};

let hawkes = MultiExponentialHawkes::new(0.5, vec![0.2], vec![1.0]);

// Simulate with fixed seed (reproducible)
let result = hawkes.simulate(100.0, Some(42));

// Simulate with entropy-based seed
let result = hawkes.simulate(100.0, None);
```

### Simulation with External Events

External events are deterministic interventions that:
- Update the process state at specified times
- Do **not** influence the random accept/reject decisions
- Enable modeling trader meta-orders or exogenous shocks

```rust
// Define external events: (time, dimension)
let meta_orders: Vec<(f64, usize)> = (0..100)
    .map(|i| (25.0 + i as f64 * 0.5, 0))  // 100 limit orders
    .collect();

let result = process.simulate_with_externals(250.0, &meta_orders, Some(42));
```

**Important:** External events affect state (and thus future intensities) but are not subject to thinning — they always occur at their specified times.

## Conditional Simulation

### Motivation

Given an observed baseline path $P$, we want to sample a **counterfactual path** $\bar{P}$ that represents "what would have happened" under different conditions (e.g., with/without trader intervention).

### `ConditionalSimulationContext`

This structure enables sampling $\bar{P} | P$ where the two paths share the same underlying randomness for certain dimensions:

```rust
pub struct ConditionalSimulationContext<'a, M: MultivariateMarkovianIntensity> {
    baseline_model: &'a M,
    counterfactual_model: M,
    baseline_path: &'a MultivariateSimulationResult,
    // ...
}
```

### Algorithm

The conditional simulator implements **independent coupling**:

1. **Shared market orders:** Dimension 2 (market orders) uses the same event times in both paths
2. **Independent limit/cancel:** Dimensions 0 and 1 are sampled independently for each path
3. **State consistency:** Both models update their Markovian state based on their respective events

```rust
use simulation::ConditionalSimulationContext;

// Baseline path (observed)
let baseline_result = baseline_model.simulate(T, Some(seed));

// Sample counterfactual conditioned on baseline
let context = ConditionalSimulationContext::new(
    &baseline_model,
    counterfactual_model,
    &baseline_result,
);

let counterfactual_result = context.simulate_conditional(T, Some(seed + 1));
```

### Use Cases

| Scenario | Baseline $P$ | Counterfactual $\bar{P}$ |
|----------|-------------|-------------------------|
| Impact of trading | No meta-orders | With meta-orders |
| Stress testing | Normal conditions | Shocked intensities |
| Strategy comparison | Strategy A | Strategy B |

## Performance Considerations

### Intensity Bounds

The thinning algorithm's efficiency depends on the tightness of $\bar{\lambda}$:

- **Too loose:** Many rejections, wasted computation
- **Too tight:** Risk of underestimating (incorrect simulation)

For Hawkes processes, the bound is computed as:

$$\bar{\lambda} = \mu + \sum_{i=1}^{k} R^i_t + \epsilon$$

where $R^i_t$ are the current Markovian factors.

### State Updates

Each accepted event requires $O(k)$ state updates for a $k$-component Hawkes kernel. The implementation minimizes allocations by updating in-place.

### Parallelization

Individual simulations are inherently sequential, but **batch simulations** parallelize perfectly:

```rust
use rayon::prelude::*;

let results: Vec<_> = (0..100)
    .into_par_iter()
    .map(|i| model.simulate(T, Some(i as u64)))
    .collect();
```

## Example: Full Workflow

```rust
use models::{AffineQueueProcess, QueueProcess, MarkovianProcessSimulator};
use simulation::ConditionalSimulationContext;

// 1. Create baseline model (no intervention)
let baseline = AffineQueueProcess::new(250, 100.0, -0.275, 2.0, 0.125,
                                        0.5, alpha.clone(), beta.clone());

// 2. Simulate baseline path
let baseline_result = baseline.simulate(250.0, Some(42));
let baseline_queue = QueueProcess::result_to_queue_path(&baseline_result, 250);

// 3. Define meta-order intervention
let meta_orders: Vec<(f64, usize)> = /* ... */;

// 4. Create counterfactual model (with intervention)
let counterfactual = AffineQueueProcess::new(/* same params */);

// 5. Sample counterfactual conditioned on baseline
let context = ConditionalSimulationContext::new(&baseline, counterfactual, &baseline_result);
let cf_result = context.simulate_conditional_with_externals(250.0, &meta_orders, Some(43));
let cf_queue = QueueProcess::result_to_queue_path(&cf_result, 250);

// 6. Compare paths for impact analysis
```
