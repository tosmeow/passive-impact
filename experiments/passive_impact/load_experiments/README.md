# Single Queue Impact

Passive market impact from a limit-order metaorder in a single-sided queue. We compare 500 counterfactual paths (gray) against the observed baseline (black), with the empirical mean in red.

## Setup

- **Queue**: $\lambda^L(q) = 100 - 0.275q$, $\lambda^C(q) = 2 + 0.125q$
- **Market orders**: Hawkes with $\alpha = [0.065, 0.2, 0.325, 0.65]$, $\beta = [0.15, 0.60, 2.5, 10.0]$
- **Metaorder**: Deterministic limit orders at rate $\nu = 5$/s for 80s
- **Impact scale**: normalized passive impact multiplied by `c_kappa_effective = -0.00001713`
- **Single-queue side convention**: `SINGLE_QUEUE_SIDE=ask` keeps the current sign; `SINGLE_QUEUE_SIDE=bid` flips impact sign for bid-posted buy limit orders
- **Paths**: 500 conditional simulations, horizon $T = 100$s

## Results

### Conditioning on the baseline queue $q$

[Impact given q](images/impact_given_q.pdf) · [Queue given q](images/queue_given_q.pdf)

*Left*: Distribution of passive impact $I(t)$ across counterfactual paths, given the observed queue $q$. *Right*: Counterfactual queue $\bar{q}$ (with metaorder) versus the baseline $q$.

### Conditioning on the impacted queue $\bar{q}$

[Impact given qbar](images/impact_given_qbar.pdf) · [Queue given qbar](images/queue_given_qbar.pdf)

*Left*: Impact distribution given the impacted queue $\bar{q}$. *Right*: Counterfactual baseline $q$ (without metaorder) versus the observed $\bar{q}$.

## How to Run

```bash
cargo run --release --bin single_queue_efficient_with_us
cargo run --release --bin single_queue_efficient_without_us
python plot_utils.py [--data-mode {general,efficient}] [--meta-end SECONDS]
```

Override the impact scale with `C_KAPPA_EFFECTIVE=<value> cargo run --release --bin ...`.
Override the single-queue side with `SINGLE_QUEUE_SIDE=bid cargo run --release --bin ...`.
`--data-mode` selects which simulation variant to plot (default: `efficient`).
`--meta-end` sets the metaorder end time drawn as a vertical line (default: `80.0`).
