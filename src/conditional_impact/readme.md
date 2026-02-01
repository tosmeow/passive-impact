# Conditional Impact

Closed-form market impact computation, under the following assumptions:

- The intensity functions $\lambda^L$ and $\lambda^C$ are such that $\lambda^L - \lambda^C$ is affine in the queue state $q$.

- Underlying price $\mathbb{E}_t[\int_0^\infty \kappa(q^a_s)dN^a_s - \kappa(q^b_s)dN^b_s]$ is built with $\kappa$ affine in the queue state.

## Structure

```
conditional_impact/
├── propagator.rs      # Hawkes kernel inversion
├── tail_intensity.rs  # Integrated conditional intensity
├── impact_factors.rs  # Tail impact at event times
└── impact_path.rs     # Full impact trajectory
```

## Impact Formula

The implementations are all done for $I(t) / c_\kappa$:

$$I(t) = c_\kappa \int_0^t (\bar{q}_s - q_s) \, dN_s + c_\kappa(\bar{q}_t - q_t) \cdot \mathcal{I}_t$$

where the tail intensity is:

$$\mathcal{I}_t = \int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds$$

## Propagator

For kernel $\varphi(s) = \sum_i \alpha_i e^{-\beta_i s}$, the propagator $\psi = \sum_j c_j e^{-\lambda_j s}$ where $\lambda_j$ are roots of:

$$1 - \sum_{i=1}^{k} \frac{\alpha_i}{x + \beta_i} = 0$$

Roots lie in intervals $(-\beta_{i+1}, -\beta_i)$; coefficients $c_j = 1/g'(\lambda_j)$.

## Tail Intensity

Closed form using the Markovian states $R^i_t = \sum_{s < t} \alpha_i e^{-\beta_i(t-s)}$ of the Hawkes process:


$$\mathcal{I}_t = C + \sum_{i=1}^{k} F_i \cdot R^i_t$$

$$C = \frac{\mu}{c_\lambda} + \mu \sum_{j=1}^{k} \frac{c_j}{\lambda_j} \left( \frac{1}{c_\lambda} - \frac{1}{\lambda_j + c_\lambda} \right)$$

$$F_i = \sum_{j=1}^{k} \frac{c_j}{\lambda_j - \beta_i} \left( \frac{1}{c_\lambda + \beta_i} - \frac{1}{c_\lambda + \lambda_j} \right) + \frac{1}{c_\lambda + \beta_i}$$

Enables O(k) evaluation at each event.

## Usage

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

## Complexity

| Operation | Complexity |
|-----------|------------|
| Propagator roots | O(k²) |
| TailImpact precompute | O(nk) |
| ImpactPath | O(n) |

Precomputation enables linear-time impact evaluation.

## The c_lambda Parameter

$c_\lambda = b_c - b_l > 0$ ensures impact decays as queue mean-reverts:
- $b_l < 0$: crowding reduces limit order arrival
- $b_c > 0$: larger queues see more cancellations
