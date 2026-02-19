# Conditional Impact

Closed-form market impact computation, under the following assumptions:

- The intensity functions $\lambda^L$ (decreasing in $q$) and $\lambda^C$ (increasing in $q$) are such that $\lambda^L - \lambda^C$ is affine in the queue state $q$, with slope $-c_{\lambda}$.

- Underlying price $\mathbb{E}_t[\int_0^\infty \kappa(q^a_s)dN^a_s - \kappa(q^b_s)dN^b_s]$ is built with $\kappa$ affine in the queue state.

At the end of the readme, we quickly explain the extension of this to incorporate two queues (bid and ask).

## Structure

```
conditional_impact/
├── impact_utils/
│   ├── propagator.rs      # Hawkes kernel inversion
│   ├── tail_intensity.rs  # Integrated conditional intensity
│   └── impact_factors.rs  # Tail impact at event times
├── single_queue/
│   └── impact_path.rs     # Full impact trajectory (single queue)
└── multi_queue/
    └── bidask_impact.rs   # Bid-ask impact (BidAskTailImpact, BidAskImpactPath)
```

## Impact Formula

The general impact formula here writes as:
$$I_t = c_\kappa \mathbb{E}_t[\int_0^\infty (\bar{q}^a_s - q^a_s)dN^a_s ]$$

We break down the blocks in our implementation as:

$$I(t) = c_\kappa \int_0^t (\bar{q}_s - q_s) \, dN_s + c_\kappa(\bar{q}_t - q_t) \cdot \mathcal{I}_t$$

where the tail intensity is:

$$\mathcal{I}_t = \int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds$$

All computations in the code are done in the case of $c_\kappa = 1$.

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

## Bid-Ask Impact

We incorporate now both queues in the intensity functions:
- $\lambda^L(q^a, q^b)$ and $\lambda^C(q^a, q^b)$ are affine functions.

- Price dynamics remain identical: Underlying price $\mathbb{E}_t[\int_0^\infty \kappa(q^a_s)dN^a_s - \kappa(q^b_s)dN^b_s]$ is built with $\kappa$ affine in the queue state.

In this case, when posting limit orders at $q^a$, we now are impacting both queues $(\bar{q}^a, \bar{q}^b)$ and the general impact form is written as:
$$I_t = c_\kappa \mathbb{E}_t[\int_0^\infty (\bar{q}^a_s - q^a_s)dN^a_s - (\bar{q}^b_s - q^b_s)dN^b_s]$$

We can write it in a similar way as the single queue case as:
$$I^x(t) = c_\kappa \int_0^t (\bar{q}^x_s - q^x_s) \, dN^x_s + c_\kappa \mathcal{I}^x_t(q_t, \bar{q}_t)$$



To make the tail term more explicit now, we need to discuss the following.

Writing $\bar{q} = (\bar{q}^a, \bar{q}^b)$, 

$$\mathbb{E}_t[\bar{q}_s - q_s] = (\bar{q}_t - q_t) - \int_t^s C_{\lambda} \mathbb{E}_t[\bar{q}_u - q_u] \, du \text{ with } C_{\lambda} = \begin{pmatrix} a_C - a_L & b_C - b_L \\\\ b_C - b_L & a_C - a_L \end{pmatrix}$$

where $\lambda^L(u,v) = a_L u + b_L v + d_L$ and $\lambda^C(u,v) = a_C u + b_C v + d_C$.

In such a scenario, there are two eigendirections $u_0 = (1,1)$ and $u_1 = (1,-1)$ with corresponding eigenvalues $c_0 = (a_C - a_L) + (b_C - b_L)$ and $c_1 = (a_C - a_L) - (b_C - b_L)$, this implies:

$$\mathbb{E}_t[\bar{q}_s - q_s] = \langle \bar{q}_t - q_t, u_0 \rangle u_0 e^{-c_0 (s-t)} + \langle \bar{q}_t - q_t, u_1 \rangle u_1 e^{-c_1 (s-t)}$$

The tail impact term for $I^x$ is now the sum of two types of tail impact identical to the single queue setting, with exponents being now $c_0$ and $c_1$.

Taking the difference $I^a - I^b$ provides then the general impact formula efficiently in the Bid - Ask queue setting.


```rust
use simulation_project::conditional_impact::{
    BidAskTailImpact, BidAskImpactPath, SymmetricCMatrix
};

let c_matrix = SymmetricCMatrix::from_affine_symmetric(
    b_l_own, b_l_cross, b_c_own, b_c_cross
);
let tail_impact = BidAskTailImpact::new_symmetric_hawkes(
    mu, alpha, beta, c_matrix, events_a, events_b
);
let impact = BidAskImpactPath::new(&q_a, &q_b, &q_prime_a, &q_prime_b, &tail_impact);
```
