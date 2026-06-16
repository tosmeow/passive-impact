# Aggressive Impact Binary

Simulates aggressive market impact under the hybrid propagator model.

## What it does

1. Pre-simulates a Hawkes process (market orders) with a fixed seed.
2. Simulates the conditioning queue path.
   - Default/with-us: condition on baseline `q` (no metaorder).
   - `--counterfactual`/without-us: condition on `bar_q` (with metaorder).
3. Injects a deterministic metaorder `N^o` (aggressive market orders that reduce the queue).
4. Runs `n_simulations` conditional simulations of the symmetric counterfactual queue:
   `bar_q` given `q` by default, or `q` given `bar_q` with `--counterfactual`.
5. Computes the aggressive market impact path using the hybrid model.
6. Writes results to `experiments/agressive_impact/load_experiments/data/`.

## Price model

$$
P_t = P_0 + \bar{\kappa}\int_0^t G(t-s)\,d(N^a-N^b)_s
    + \int_0^t(\kappa(q^a_s)-\bar{\kappa})\,dN^a_s
    - \int_0^t(\kappa(q^b_s)-\bar{\kappa})\,dN^b_s.
$$

## Impact formula

$$
MI_t = \bar{\kappa}\int_0^t G(t-s)\,dN^{o,a}_s
     + \int_0^t [\kappa(\bar{q}^a_s)-\kappa(q^a_s)]\,dN^a_s.
$$

## Parameters

- Queue: $\lambda^L(q) = 100 - 0.275q$, $\lambda^C(q) = 2 + 0.125q$
- Hawkes: $\mu = 1$, $\alpha = [0.065, 0.2, 0.325, 0.65]$, $\beta = [0.15, 0.60, 2.5, 10.0]$
- Metaorder: 156 events, $t \in [0, 60]$
- $\kappa(q) = -0.001q$
- $\bar{\kappa} = 0.01$

## Output files

Written to `experiments/agressive_impact/load_experiments/data/with/`.
The `--counterfactual` flag writes the symmetric data to
`experiments/agressive_impact/load_experiments/data/without/`.

| File | Shape | Contents |
|---|---|---|
| `impact_paths.npy` | `(n_times, n_sims)` | $MI(t)$ per simulation |
| `queue_paths.npy` | `(n_times, n_sims + 1)` | With-us: first col = $q$, remaining = $\bar{q}$ per simulation. Without-us: first col = $\bar{q}$, remaining = $q$ per simulation |
| `times.npy` | `(n_times,)` | Evaluation times (merged market + metaorder times) |
| `event_types.npy` | `(n_times,)` | 1.0 = market order, 0.0 = metaorder |
| `bar_kappa.npy` | `(1,)` | Constant propagated metaorder weight |

## Visualization

```bash
cargo run --release --bin agressive_impact
cargo run --release --bin agressive_impact -- --counterfactual
python experiments/agressive_impact/load_experiments/plot_utils.py
python experiments/agressive_impact/load_experiments/plot_utils.py --counterfactual
```
