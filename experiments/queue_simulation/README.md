# Queue Simulation

Counterfactual **queue paths** under a metaorder, without computing the price-impact curve. Same simulator as `passive_impact` (conditional thinning of a queue-reactive process driven by a multi-exponential Hawkes), but stops short of evaluating $I(t)$ — useful when you want to study queue dynamics in isolation.

For each Monte Carlo path you get the realised counterfactual queue size $\bar{q}(t)$ on a uniform time grid, alongside the baseline $q(t)$ from the conditioning trajectory.

## Custom experiment — `custom_experiment/`

Edit the `config` block at the top of [`custom_experiment/main.py`](custom_experiment/main.py) and run:

```bash
python experiments/queue_simulation/custom_experiment/main.py
```

Knobs: `time_horizon`, `n_simulations`, `n_eval_times` (size of the uniform sampling grid), `initial_queue_size`, `mode` (`"single"` | `"double"`), the Hawkes parameters, the affine-queue parameters, and the metaorder shape (integer count or explicit list of arrival times).

Set `counterfactual=False` for with-us conditioning, or `counterfactual=True` for the without-us counterfactual. Outputs (`times.npy`, `queue_paths.npy` with shape `(n_eval_times, n_simulations + 1)`) land in `custom_experiment/output/` (gitignored).

## Pre-saved baselines — `load_experiments/`

The notebook [`analysis.ipynb`](load_experiments/analysis.ipynb) loads `.npy` baselines from `load_experiments/data/single/efficient/` and renders the queue-shade plot via [`plot_utils.py`](load_experiments/plot_utils.py).

Baseline `.npy` data is gitignored — regenerate via the Rust binary:

```bash
cargo run --release --bin queue_simulation_efficient
```
