# Aggressive Impact

Market-order impact under the hybrid aggressive-impact model. A buy-side
metaorder consumes ask liquidity; metaorder flow is propagated through the
Hawkes-derived kernel with constant weight $\bar{\kappa}$, while ordinary
market-order queue corrections enter instantaneously through
$\kappa(\bar{q}) - \kappa(q)$.

## Setup

- **Price model**: hybrid propagator plus instantaneous queue correction.
- **Default correction**: $\kappa(q) = -0.001q$.
- **Default propagated metaorder weight**: $\bar{\kappa}=0.01$.
- **Metaorder**: 156 market-order events in the Rust baseline.
- **Paths**: 500 conditional simulations, horizon $T = 90$.

## Results

### Impact trajectory

[Impact paths given q](images/impact_paths_given_q.png)

Distribution of aggressive impact $MI(t)$ across 500 counterfactual paths
(gray), with mean in red.

### Queue dynamics

[Queue paths given q](images/queue_paths_given_q.png)

Counterfactual queue $\bar{q}$ (with metaorder, gray paths) versus baseline
$q$ (black).

## How to Run

```bash
cargo run --release --bin agressive_impact
cargo run --release --bin agressive_impact -- --counterfactual
python plot_utils.py
python plot_utils.py --counterfactual
python plot_utils.py --counterfactual \
  --data-base ../custom_experiment/output/without_us
```

By default, `plot_utils.py` generates both conditioning cases. The legacy
`--counterfactual` flag is an alias for `--scenario without`: those plots read
the first queue column as `bar_q` and the simulations as `q_sim_*`, while the
with-us case reads `q` and `bar_q_sim_*`. Generated image names end in
`_given_q.png` for the default with-us conditioning and `_given_qbar.png` for
the counterfactual without-us conditioning. The canonical aggressive plots are
`impact_paths_*.png` and `queue_paths_*.png`.
