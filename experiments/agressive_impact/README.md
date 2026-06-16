# Aggressive Impact

Market impact when the metaorder is **aggressive** (consumes liquidity via
market orders, dim=2). The experiment uses the hybrid aggressive-impact model:
metaorder flow is propagated through the Hawkes-derived kernel with constant
weight $\bar{\kappa}$, while ordinary market-order queue corrections use
$\kappa(\bar{q}) - \kappa(q)$ instantaneously.

## Custom experiment — `custom_experiment/`

Edit the `config` block at the top of [`custom_experiment/main.py`](custom_experiment/main.py) and run:

```bash
python experiments/agressive_impact/custom_experiment/main.py
```

Knobs: `time_horizon`, `n_simulations`, `initial_queue_size`,
`counterfactual` (`False` for with-us, `True` for without-us), `bar_kappa`, the
Hawkes parameters, the affine-queue parameters, the metaorder shape, and a
`kappa: Callable[[float], float]` for the instantaneous queue correction
(defaults to $\kappa(q) = -0.001q$).

Outputs (`times.npy`, `queue_paths.npy`, `impact_paths.npy`, `event_types.npy`,
`bar_kappa.npy`) land in `custom_experiment/output/with_us/` or
`custom_experiment/output/without_us/` (gitignored).

## Pre-saved baselines — `load_experiments/`

The notebook [`analysis.ipynb`](load_experiments/analysis.ipynb) loads `.npy`
baselines from `load_experiments/data/` and reproduces the standard plots via
[`plot_utils.py`](load_experiments/plot_utils.py). Pass `--counterfactual` or
`--scenario without` when plotting only without-us outputs so the first queue
column is read as `bar_q` and the simulations as `q_sim_*`.

Baseline `.npy` data is gitignored — regenerate via the Rust binary:

```bash
cargo run --release --bin agressive_impact
cargo run --release --bin agressive_impact -- --counterfactual
```

The default cargo run writes the with-us data under
`load_experiments/data/with/`. The `--counterfactual` flag writes the symmetric
without-us data under `load_experiments/data/without/`. After both data
directions exist, run:

```bash
python experiments/agressive_impact/load_experiments/plot_utils.py
```

to generate `impact_paths_given_q.pdf`, `queue_paths_given_q.pdf`,
`impact_paths_given_qbar.pdf`, and `queue_paths_given_qbar.pdf` under
`load_experiments/images/`. Pass `--format png` if PNG files are needed for a
downstream workflow.
