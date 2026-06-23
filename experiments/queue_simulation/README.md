# Queue Simulation

Counterfactual **queue paths** under a metaorder, without computing the price-impact curve. Same simulator as `passive_impact` (conditional thinning of a queue-reactive process driven by a multi-exponential Hawkes), but stops short of evaluating $I(t)$ — useful when you want to study queue dynamics in isolation.

For each Monte Carlo path you get the realised counterfactual queue size $\bar{q}(t)$ on a uniform time grid, alongside the baseline $q(t)$ from the conditioning trajectory.

## Custom experiment — `custom_experiment/`

Edit the `config` block at the top of [`custom_experiment/main.py`](custom_experiment/main.py) and run:

```bash
python experiments/queue_simulation/custom_experiment/main.py
```

Knobs: `time_horizon`, `n_simulations`, `n_eval_times` (size of the uniform sampling grid), `initial_queue_size`, `mode` (`"single"` | `"double"`), the Hawkes parameters, the affine-queue parameters, and the metaorder shape (integer count or explicit list of arrival times). Double mode additionally accepts `initial_ask_queue_size`, `initial_bid_queue_size`, `metaorder_side`, `b_l_cross`, and `b_c_cross`.

Set `counterfactual=False` for with-us conditioning, or `counterfactual=True` for the without-us counterfactual. Outputs land in `custom_experiment/output/with_us/` or `custom_experiment/output/without_us/` (gitignored). Single mode writes `times.npy` and `queue_paths.npy`; double mode writes `times.npy`, `ask_queue_paths.npy`, and `bid_queue_paths.npy`.

## Pre-saved baselines — `load_experiments/`

The notebook [`analysis.ipynb`](load_experiments/analysis.ipynb) loads `.npy` baselines from `load_experiments/data/single/efficient/` and renders the queue-shade plot via [`plot_utils.py`](load_experiments/plot_utils.py). Pass `--counterfactual` when plotting without-us outputs so the first queue column is read as `bar_q` and the simulations as `q_sim_*`.

Baseline `.npy` data is gitignored — regenerate via the Rust binary:

```bash
cargo run --release --bin queue_simulation_efficient
```
