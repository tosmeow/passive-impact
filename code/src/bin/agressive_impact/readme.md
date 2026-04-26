# Aggressive Impact Binary

Simulates aggressive market impact under the propagator price model.

## What it does

1. Pre-simulates a Hawkes process (market orders) with a fixed seed.
2. Simulates a baseline queue path `q` (no metaorder).
3. Injects a deterministic metaorder `N^o` (aggressive market orders that reduce the queue).
4. Runs `n_simulations` conditional simulations of the counterfactual queue `bar_q` given the observed `q`, with the metaorder present.
5. Computes the aggressive market impact path at each simulation using `AggressiveImpactPath`.
6. Writes results to `data/agressive_impact/`.

## Price model

$$P_t = P_0 + \int_0^t \kappa(q^a_s)\, G(t-s)\, dN^a_s - \int_0^t \kappa(q^b_s)\, G(t-s)\, dN^b_s$$

where $G$ is the martingale-consistent propagator kernel derived from the Hawkes kernel.

## Impact formula

$$MI_t = \int_0^t [\kappa(\bar{q}^a_s) - \kappa(q^a_s)]\, G(t-s)\, dN^a_s + \int_0^t \kappa(\bar{q}^a_s)\, G(t-s)\, dN^{o,a}_s$$

## Parameters (matching single queue experiment)

- Queue: $\lambda^L(q) = 100 - 0.275q$, $\lambda^C(q) = 2 + 0.125q$
- Hawkes: $\mu = 1$, $\alpha = [0.065, 0.2, 0.325, 0.65]$, $\beta = [0.15, 0.60, 2.5, 10.0]$
- Metaorder: 200 events, $t \in [1, 75]$
- $\kappa(q) = c_1 \sqrt{\log(e^{-c_2 q} + 1)}$ with $c_1 = 1000$, $c_2 = 0.01$

## Output files

Written to `data/agressive_impact/`:

| File | Shape | Contents |
|---|---|---|
| `impact_paths.npy` | `(n_times, n_sims)` | $MI(t)$ per simulation |
| `queue_paths.npy` | `(n_times, n_sims + 1)` | First col = $q$; remaining = $\bar{q}$ per simulation |
| `times.npy` | `(n_times,)` | Evaluation times (merged market + metaorder times) |
| `event_types.npy` | `(n_times,)` | 1.0 = market order, 0.0 = metaorder |

## Visualization

```bash
cd python/experiments/agressive_impact
python plot_utils.py
```
