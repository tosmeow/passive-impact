# Aggressive Impact

Market-order impact under the propagator price model. A buy-side metaorder consumes ask liquidity; the price responds through the propagator kernel $G(t) = 1 + \sum_i \frac{\alpha_i/\beta_i}{1-\|\varphi\|_1} e^{-\beta_i t}$.

## Setup

- **Price model**: $P_t = \int_0^t \kappa(q^a_s) G(t-s) dN^a_s - \int_0^t \kappa(q^b_s) G(t-s) dN^b_s$
- **Impact function**: $\kappa(q) = c_1 \sqrt{\log(e^{-c_2 q} + 1)}$ with $c_1 = 1000$, $c_2 = 0.01$ (square-root behaviour for large $q$)
- **Propagator**: $G(0) = 1/(1-\|\varphi\|_1) \approx 26$ (mean cluster size), $G(\infty) = 1$
- **Metaorder**: 200 market-order events
- **Paths**: 500 conditional simulations, horizon $T = 100$.

## Results

### Impact trajectory

[Impact paths given q](images/impact_paths_given_q.pdf)

Distribution of aggressive impact $MI(t)$ across 500 counterfactual paths (gray), with mean in red.

### Queue dynamics

[Queue paths given q](images/queue_paths_given_q.pdf)

Counterfactual queue $\bar{q}$ (with metaorder, gray paths) versus baseline $q$ (black).

## How to Run

```bash
cargo run --release --bin agressive_impact
cargo run --release --bin agressive_impact -- --counterfactual
python plot_utils.py
python plot_utils.py --model propagator --counterfactual
python plot_utils.py --model propagator --counterfactual \
  --data-base ../custom_experiment/output/without_us
```

By default, `plot_utils.py` generates both conditioning cases. The legacy
`--counterfactual` flag is an alias for `--scenario without`: those plots read
the first queue column as `bar_q` and the simulations as `q_sim_*`, while the
with-us case reads `q` and `bar_q_sim_*`. Generated image names end in
`_given_q.pdf` for the default with-us conditioning and `_given_qbar.pdf` for
the counterfactual without-us conditioning. The canonical aggressive plots are
`impact_paths_*.pdf` and `queue_paths_*.pdf`.
