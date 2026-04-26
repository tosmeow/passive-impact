# Aggressive Impact

Market impact when the metaorder is **aggressive** (consumes liquidity via market orders, dim=2). The per-trade impact is shaped by a queue-dependent kernel $\kappa(q)$, and the price path is a martingale propagator over the Hawkes-driven trade flow. Two model variants share the same simulator but differ in how the per-trade impact is composed:

- **Propagator** — pure propagator: $\kappa(q)$ is the only impact term per trade.
- **Hybrid** — combines a queue-dependent instantaneous correction $\kappa(q) = -c_\kappa \cdot q$ with a propagator term weighted by a constant $\bar{\kappa}$.

## Custom experiment — `custom_experiment/`

Edit the `config` block at the top of [`custom_experiment/main.py`](custom_experiment/main.py) and run:

```bash
python experiments/agressive_impact/custom_experiment/main.py
```

Knobs: `time_horizon`, `n_simulations`, `initial_queue_size`, `model` (`"propagator"` | `"hybrid"`), `bar_kappa` (required for hybrid), the Hawkes parameters, the affine-queue parameters, the metaorder shape, and a `kappa: Callable[[float], float]` for the per-trade impact function (defaults to the paper's $c_1 \sqrt{\log(e^{-c_2 q} + 1)}$).

Outputs (`times.npy`, `queue_paths.npy`, `impact_paths.npy`, `event_types.npy`) land in `custom_experiment/output/` (gitignored).

## Pre-saved baselines — `load_experiments/`

The notebook [`analysis.ipynb`](load_experiments/analysis.ipynb) loads `.npy` baselines from `load_experiments/data/{propagator,hybrid}/` and reproduces the standard plots via [`plot_utils.py`](load_experiments/plot_utils.py) (which dispatches to `plot_utils_propagator.py` and `plot_utils_hybrid.py`).

Baseline `.npy` data is gitignored — regenerate via the Rust binaries:

```bash
cargo run --release --bin agressive_impact          # propagator model
cargo run --release --bin agressive_impact_hybrid   # hybrid model
```
