# Passive Market Impact Simulation

A high-performance Rust library for simulating and analyzing market impact using point processes. Combines Hawkes processes (for market orders) with queue-reactive Markovian dynamics (for limit orders and cancellations) to compute the price effect of trading strategies through conditional path simulation. This comes accompanying our paper **Conditional Simulation of Poisson Measures and Market Impact** [link].

## Visual Overview

<p align="center">
  <img src="experiments/passive_impact/load_experiments/images/impact_given_q.png" width="48%" alt="Conditional impact distribution given baseline queue"/>
  <img src="experiments/passive_impact/load_experiments/images/queue_given_q.png" width="48%" alt="Conditional queue distribution given baseline"/>
</p>
<p align="center">
  <img src="experiments/passive_impact/load_experiments/images/impact_given_qbar.png" width="48%" alt="Impact given shocked queue"/>
  <img src="experiments/passive_impact/load_experiments/images/queue_given_qbar.png" width="48%" alt="Queue given shocked queue"/>
</p>

*Conditional simulation of 500 counterfactual market paths (gray shading) with empirical mean (red) and observed baseline (black). Each panel shows a different initial queue state.*

## What This Library Provides


- **Exact conditional simulation of Poisson processes** given an observed initial trajectory, by either adding or removing jump times consistently with the conditioning.
- **Counterfactual queue simulation.** Given an observed queue trajectory, simulate what the queue would have been under a different metaorder scenario:
  - *Removing a metaorder.* Starting from an observed trajectory that contains a metaorder, simulate the counterfactual queue in which the metaorder was never sent.
  - *Adding a metaorder.* Starting from an observed (baseline) trajectory, simulate the counterfactual queue in which an additional buy/sell metaorder is injected, executed as either limit or market orders.
- **Conditional market impact.** Compare observed and counterfactual queues to recover the price impact attributable to a metaorder — covering both passive (limit-order) and aggressive (market-order) impact. This enables two complementary analyses on the *same* observed trajectory:
  - *Ex-post (a posteriori) impact of an executed strategy.* For a trading strategy that was actually executed, recover the full **distribution** of its market impact conditional on the observed market data.
  - *Impact of alternative strategies.* On the same observed trajectory, estimate the impact that a different, hypothetical strategy would have had — enabling realistic backtesting and side-by-side comparison of strategies against the same realised market conditions.
- **Closed-form market impact** under a Hawkes market-order flow with a sum-of-exponentials kernel, enabling impact estimation without nested Monte Carlo.
- **Impact-cost experiments** that replay from real queue snapshots and own order postings with the associated execution flags.
- **Flexible architecture** supporting both single-queue and bid-ask queue-pair scenarios, with optimized ("efficient") and general simulation variants.

## Setup

Fresh-clone workflow using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/tosmeow/passive-impact.git
cd passive-impact

uv venv
source .venv/bin/activate
unset CONDA_PREFIX
uv pip install -e ".[dev]"
maturin develop --release --manifest-path code/python/Cargo.toml
```

Notes on the steps:
- `unset CONDA_PREFIX` is needed only if your shell auto-activates a conda env (you'll see `(base)` in your prompt). Maturin refuses to build when both `VIRTUAL_ENV` and `CONDA_PREFIX` are set.
- `uv pip install -e ".[dev]"` installs the dev tooling (maturin, pytest, jupyter, ipykernel, nbconvert).
- On Python ≥3.13, prefix the `maturin develop` command with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.

Verify the install (the full Python suite should pass, then prints `0.1.0`):

```bash
pytest code/python/tests/
python -c "import simproj; print(simproj.__version__)"
```

> **Note:** `cargo build` from the repo root will also try to build the bindings crate. On Python ≥3.13, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` for that path too, or build the lib in isolation with `cargo build -p simulation_project`.

## Quick Start (Python)

After [Setup](#setup), run any of the three simulation demo categories from Python — each ships a `custom_experiment/main.py` whose top section is a config dataclass. Edit the config, run the file, and `.npy` outputs land in `output/with_us/` or `output/without_us/` according to `counterfactual`.

```python
# experiments/passive_impact/custom_experiment/main.py
from simproj import passive_impact as pi

config = pi.PassiveImpactConfig(
    time_horizon=100.0,
    n_simulations=500,
    initial_queue_size=200,
    mode="single",                 # "single" | "double"
    counterfactual=False,          # False: with us | True: without us
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    # Metaorder accepts: int N (evenly spaced inside metaorder_window)
    #                    list/ndarray (explicit arrival times — window ignored)
    metaorder=375,
    metaorder_window=(1.0, 80.0),
    seed=42,
)

result = pi.run(config)               # dict[str, np.ndarray]
direction = "without_us" if config.counterfactual else "with_us"
pi.save(result, f"experiments/passive_impact/custom_experiment/output/{direction}/")
```

Run it:

```bash
python experiments/passive_impact/custom_experiment/main.py
```

The same shape applies to `agressive_impact` and `queue_simulation` — each has its own config dataclass (`AggressiveImpactConfig`, `QueueSimulationConfig`) and `main.py` template. The empirical passive execution-cost workflow lives under [`experiments/impact_cost/`](experiments/impact_cost/) and is driven by the canonical lifecycle config in `load_experiments/config.toml`.


## Mathematical Background

### Hawkes Process

A Hawkes process is a self-exciting Poisson process with intensity

$$\lambda_t = \mu + \int_0^{t-} \phi(t-s)dN_s$$

where $\mu$ is the baseline exogeneous intensity and $\phi$ is the nonincreasing self-exciting kernel. In the current framework, we will mostly used sum-of-exponentials kernel of the form

$$ \lambda_t = \mu + \sum_{i=1}^{k} R^i_t, \quad \varphi(s) = \sum_{i=1}^{k} \alpha_i e^{-\beta_i s}$$

with Markovian states $R^i_t := \int_0^{t-} \alpha_i e^{-\beta_i(t-s)} dN_s$ enabling O(1) intensity updates.

Note that power-law kernels can be accurately approximated by sums of exponentials, so the framework extends to this class of kernels as well.


### Aggregate Queue Dynamics

The aggregate queues $q^a$ and $q^b$ are given by

$$q^a = L^a - C^a - N^a, \qquad \text{and} \qquad q^b = L^b - C^b - N^b$$

where for $x \in \{a,b\}$, $L^{x}$ and $C^{x}$ are point processes representing the limit orders and cancellations with state-dependent intensities

$$\lambda^{L,x}(q) = a_l + b_l \cdot q, \qquad \lambda^{C,x}(q) = a_c + b_c \cdot q$$

with $b_l < 0$ and $b_c > 0$.

For market orders, $N^a$ and $N^b$ are modeled using two Hawkes process with the same baseline intensity and same kernel.


### Price Process

The price is defined as the anticipation of future order flow, while allowing the contribution of each market order to depend on available liquidity. 

$$
P_t=P_0+\lim_{T\to\infty}\mathbb{E} \left[\int_0^T \kappa(q^a_s) \mathrm{d}N^a_s-\int_0^T \kappa(q^b_s) \mathrm{d}N^b_s\ \middle|\ \mathcal F_t\right].
$$

for a given impact function $\kappa$.


### Conditional Impact

If one executes a given metaorder $X^o$ at the ask side, the observed dynamics for the ask queue and price are

$$ \overline{q}^{a,t}_s = q_0^a+\overline{L}^{a,t}_s-\overline{C}^{a,t}_s-N^a_s+X_s^{o,t}, \qquad s\ge 0, \qquad \text{and} \qquad \overline{P}_t = P_0+\lim_{T\to\infty}\mathbb{E}\left[\int_0^T \kappa(\overline{q}^{a,t}_s) \mathrm{d}N^a_s - \int_0^T \kappa(q^b_s) \mathrm{d}N^b_s \ \middle|\ \mathcal F_t\right]$$

where $X_s^{o,t}:=X_{s \wedge t}^{o}, s \geq 0$ can be understood as a sequence of limit orders ($X^o = L^o$) or market orders ($X^o = -N^o$).

The impact is then given by:

$$ MI(t) = \overline{P}_t  - P_t$$



### Passive Market Impact

Accurately simulating price impact reduces to simulating the counterfactual price trajectory — or, more concretely, the counterfactual queue trajectories that would have been observed had the metaorder not been sent. This is what our framework achieves — see the paper for the theory and the code below for the implementation. Thus, under the following specifications:
- Passive metaorder $L^o$ executed exclusively using limit orders ($X = L^o$),
- Affine intensities $\lambda^L$ and $\lambda^C$,
- Affine impact function $\kappa(q) = c_\kappa \dot q + d_\kappa$,
- Sum-of-exponentials Hawkes kernel,

we obtain a closed formula for the distribution of passive market impact, and we implement:

$$\mathrm{MI}(t) = c_\kappa \int_0^t (\overline{q}_s - q_s) \mathrm{d}N_s + c_\kappa (\overline{q}_t - q_t) \cdot \mathrm{MI}_t$$

thus providing the following solution relying on the resolvent operator $(\delta_0 - \varphi)^{-1}$:

```math
\mathrm{MI}_t = c_{\kappa} \int_0^t (\overline{q}^{a,t}_s - q^{a}_s)\,\mathrm{d} N^a_s
+ c_{\kappa} (\overline{q}^{a,t}_t - q^{a}_t)\Big(\zeta + \int_0^t \sum_{i=1}^m \gamma_i e^{-\beta_i (t-s)}\,\mathrm{d}N^a_s\Big).
```
for some constants $(\gamma_i)_{1\le i\le m}$ and $\zeta$.

The config-level mapping is documented in
[`experiments/impact_cost/load_experiments/FORMULAS.md`](experiments/impact_cost/load_experiments/FORMULAS.md).

### Aggressive Market Impact

For aggressive metaorders, the strategy consumes liquidity directly through
market orders. The hybrid aggressive-impact model in `conditional_impact`
compares the impacted queue path against the no-metaorder counterfactual,
propagates deterministic metaorder flow with constant `bar_kappa`, and adds the
queue-dependent market-order correction instantaneously.

## Modules

| Module | Description |
|--------|-------------|
| [`models`](code/src/models/) | Hawkes, queues, Markovian abstractions |
| [`simulation`](code/src/simulation/) | Thinning algorithm, conditional simulation |
| [`simulation_helpers`](code/src/simulation_helpers/) | Parallel batch simulation, event utilities |
| [`conditional_impact`](code/src/conditional_impact/) | Resolvent and propagator impact models |
| [`experiments::impact_cost`](code/src/experiments/impact_cost/) | Experiment-scoped native helpers for anchored empirical queues and passive execution fills |
| [`utils`](code/src/utils/) | IVT root-finding, finite differences |

## Experiments

Four top-level experiment categories live under `experiments/`:

| Category | What it shows | Entry points |
|---|---|---|
| **Passive Impact** | Conditional impact from limit-order metaorders (single + double queue) | [`load_experiments/analysis.ipynb`](experiments/passive_impact/load_experiments/analysis.ipynb), [`custom_experiment/main.py`](experiments/passive_impact/custom_experiment/main.py) |
| **Aggressive Impact** | Hybrid market-order impact from aggressive metaorders | [`load_experiments/analysis.ipynb`](experiments/agressive_impact/load_experiments/analysis.ipynb), [`custom_experiment/main.py`](experiments/agressive_impact/custom_experiment/main.py) |
| **Queue Simulation** | Counterfactual queue paths under a metaorder (no impact curve) | [`load_experiments/analysis.ipynb`](experiments/queue_simulation/load_experiments/analysis.ipynb), [`custom_experiment/main.py`](experiments/queue_simulation/custom_experiment/main.py) |
| **Impact Cost** | Empirical passive lifecycle execution-cost workflow using aggregate queue snapshots and tail-propagator price impact | [`README.md`](experiments/impact_cost/README.md), [`COMPONENTS.md`](experiments/impact_cost/COMPONENTS.md), [`load_experiments/`](experiments/impact_cost/load_experiments/) |

For the older simulation experiments, each `load_experiments/` folder contains
the notebook, plot utilities, and a `data/` subtree where the Rust binaries
write `.npy` files. The impact-cost `load_experiments/` folder instead uses a
single `config.toml`, CSV outputs, and figure utilities. Generated data is
gitignored unless explicitly promoted. Each `custom_experiment/` folder
contains a single `main.py` whose top section is a config dataclass; edit and
run.

### Run a custom experiment (Python — primary user path)

Edit the config block at the top of the relevant `main.py`, then run it:

```bash
python experiments/passive_impact/custom_experiment/main.py
python experiments/agressive_impact/custom_experiment/main.py
python experiments/queue_simulation/custom_experiment/main.py
```

Outputs (`.npy` arrays) land in each folder's `output/with_us/` or `output/without_us/` directory (gitignored). The Python facades produce the same shapes as the Rust binaries — see each `<Category>Config` dataclass in [`code/python/simproj/`](code/python/simproj/) for the full set of knobs (Hawkes parameters, queue parameters, metaorder shape, `bar_kappa`, etc.).

### Run the impact-cost workflow

The canonical impact-cost experiment now lives under
`experiments/impact_cost/load_experiments/`. Edit its `config.toml`, then run
the lifecycle workflow:

```bash
python -m experiments.impact_cost.load_experiments.lifecycle_passive_cost \
  --config experiments/impact_cost/load_experiments/config.toml
```

This lifecycle path is fixed to the tail-propagator impact model. CSV/JSON
outputs land under `load_experiments/data/`; PNG figures land under
`load_experiments/images/`. Older raw-fill inference and diagnostic scripts are
kept under `experiments/impact_cost/archive/diagnostics/`.

Start with [`experiments/impact_cost/README.md`](experiments/impact_cost/README.md) for the queue/price-sign conventions and [`experiments/impact_cost/COMPONENTS.md`](experiments/impact_cost/COMPONENTS.md) for the component map.

### Inspect pre-saved baselines (notebooks)

Each category's `load_experiments/analysis.ipynb` loads `.npy` data from its sibling `data/` directory and renders the standard plots. The `.npy` files are gitignored — regenerate them via the Rust binaries below before opening the notebook on a fresh checkout.

### Regenerate baselines (Rust binaries)

The original Rust binaries still exist for fast batch baseline generation:

    cargo run --release --bin single_queue_efficient_with_us
    cargo run --release --bin single_queue_efficient_without_us
    cargo run --release --bin double_queue_efficient_with_us
    cargo run --release --bin double_queue_efficient_without_us
    cargo run --release --bin agressive_impact
    cargo run --release --bin queue_simulation_efficient

(General variants are also kept for validation: `*_general_with_us`, `*_general_without_us`.)

---

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `scipy`, `pyarrow`: Python facades, plotting, and parquet-backed experiment inputs
- `maturin`, `pyo3`: Rust-to-Python bindings (development only)
- `pytest`: Testing framework
