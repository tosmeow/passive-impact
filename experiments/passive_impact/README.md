# Passive Impact

Conditional market impact from **limit-order metaorders** in a queue-reactive market — the metaorder enters the limit/cancel queue, and the resulting price impact is computed in closed form via the resolvent operator on a multi-exponential Hawkes kernel. Supports both single-queue and bid-ask (double) queue setups.

The math:

$$I(t) = c_\kappa \int_0^t (\bar{q}_s - q_s) \, dN_s + c_\kappa (\bar{q}_t - q_t) \cdot \mathcal{I}_t$$

with $\mathcal{I}_t = \int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds$ (closed form via $(\delta_0 - \varphi)^{-1}$).

## Custom experiment — `custom_experiment/`

Edit the `config` block at the top of [`custom_experiment/main.py`](custom_experiment/main.py) and run:

```bash
python experiments/passive_impact/custom_experiment/main.py
```

Knobs: `time_horizon`, `n_simulations`, `initial_queue_size`, `mode` (`"single"` | `"double"`), `counterfactual` (`False` for with-us, `True` for without-us), the Hawkes parameters `mu`/`alpha`/`beta`, the affine-queue parameters `a_l`/`b_l`/`a_c`/`b_c`, and the metaorder shape (either an integer count for evenly-spaced orders inside `metaorder_window`, or an explicit `list`/`np.ndarray` of arrival times).

Outputs (`times.npy`, `queue_paths.npy`, `impact_paths.npy`) land in `custom_experiment/output/` (gitignored).

## Pre-saved baselines — `load_experiments/`

The two notebooks ([`analysis.ipynb`](load_experiments/analysis.ipynb), [`analysis_double.ipynb`](load_experiments/analysis_double.ipynb)) load committed `.npy` baselines from `load_experiments/data/{single,double}/{efficient,general}/{with,without}/` and reproduce the standard plots via [`plot_utils.py`](load_experiments/plot_utils.py) (which dispatches to `plot_utils_single.py` and `plot_utils_double.py`).

Baseline `.npy` data is gitignored — regenerate it via the Rust binaries before opening the notebooks on a fresh checkout:

```bash
cargo run --release --bin single_queue_efficient_with_us
cargo run --release --bin single_queue_efficient_without_us
cargo run --release --bin double_queue_efficient_with_us
cargo run --release --bin double_queue_efficient_without_us
```
