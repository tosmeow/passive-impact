# Passive Market Impact Simulation

A high-performance Rust library for simulating and analyzing market impact using point processes. Combines Hawkes processes (for market orders) with queue-reactive Markovian dynamics (for limit orders and cancellations) to compute the price effect of trading strategies through conditional path simulation.

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

- **Exact conditional simulation** of coupled point processes, enabling pathwise comparison of observed vs. counterfactual market scenarios.
- **Queue-reactive order dynamics** with affine intensity functions, modeling how limit orders, cancellations, and market orders respond to queue depth.
- **Closed-form market impact computation** for Hawkes with kernels as sum of exponentials using resolvant operator methods, enabling efficient impact estimation without nested Monte Carlo.
- **Flexible architecture** supporting both single-queue and bid-ask queue pair scenarios, with optimized ("efficient") and general simulation variants.

## Setup

Fresh-clone workflow using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/tosmeow/passive-impact.git
cd passive-impact

uv venv                             # create .venv/
source .venv/bin/activate           # activate it
uv pip install -e ".[dev]"          # installs maturin, pytest, jupyter, ipykernel, nbconvert

# Build the Rust bindings (simproj package)
cd code/python && maturin develop --release && cd -
# On Python ≥3.13, prefix the maturin command with PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

Verify the install:

```bash
pytest code/python/tests/           # 14 smoke tests
python -c "import simproj; print(simproj.__version__)"
```

> **Note:** `cargo build` from the repo root will also try to build the bindings crate. On Python ≥3.13, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` for that path too, or build the lib in isolation with `cargo build -p simulation_project`.

## Quick Start (Python)

After [Setup](#setup), run any of the three experiment categories from Python — each ships a `custom_experiment/main.py` whose top section is a config dataclass. Edit the config, run the file, and `.npy` outputs land in `output/`.

```python
# experiments/passive_impact/custom_experiment/main.py
from simproj import passive_impact as pi

config = pi.PassiveImpactConfig(
    time_horizon=100.0,
    n_simulations=500,
    initial_queue_size=200,
    mode="single",                 # "single" | "double"
    side="both",                   # "with" | "without" | "both"
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
pi.save(result, "experiments/passive_impact/custom_experiment/output/")
```

Run it:

```bash
python experiments/passive_impact/custom_experiment/main.py
```

The same shape applies to `agressive_impact` and `queue_simulation` — each has its own config dataclass (`AggressiveImpactConfig`, `QueueSimulationConfig`) and `main.py` template. See [Experiments](#experiments) below for the full layout.

## Mathematical Background

### Hawkes Process

Conditional intensity with multi-exponential kernel:

$$\lambda_t = \mu + \int_0^{t-} \phi(t-s)dN_s = \mu + \sum_{i=1}^{k} R^i_t, \quad \varphi(s) = \sum_{i=1}^{k} \alpha_i e^{-\beta_i s}$$

with Markovian states $R^i_t := \int_0^{t-} \alpha_i e^{-\beta_i(t-s)} dN_s$ enabling O(1) intensity updates.

### Queue Dynamics

- **Limits**: $\lambda^L(q) = a_l + b_l \cdot q$, with $b_l < 0$.
- **Cancels**: $\lambda^C(q) = a_c + b_c \cdot q$ with $b_c > 0$.
- **Markets**: Hawkes process.

### Conditional Impact

With $c_\lambda := b_c - b_l$, we implement:

$$I(t) = c_\kappa \int_0^t (\bar{q}_s - q_s) \, dN_s + c_\kappa (\bar{q}_t - q_t) \cdot \mathcal{I}_t$$

where the following term admits a closed form relying on the resolvent operator $(\delta_0 - \varphi)^{-1}$:
```math
\mathcal{I}_t = \int_t^\infty e^{-c_\lambda(s-t)} \mathbb{E}_t[\lambda_s] \, ds
```

## Modules

| Module | Description |
|--------|-------------|
| [`models`](code/src/models/) | Hawkes, queues, Markovian abstractions |
| [`simulation`](code/src/simulation/) | Thinning algorithm, conditional simulation |
| [`simulation_helpers`](code/src/simulation_helpers/) | Parallel batch simulation, event utilities |
| [`conditional_impact`](code/src/conditional_impact/) | Resolvent and propagator impact models |
| [`utils`](code/src/utils/) | IVT root-finding, finite differences |

## Experiments

Three top-level experiment categories live under `experiments/`:

| Category | What it shows | Pre-saved baseline | Custom |
|---|---|---|---|
| **Passive Impact** | Conditional impact from limit-order metaorders (single + double queue) | [`experiments/passive_impact/load_experiments/analysis.ipynb`](experiments/passive_impact/load_experiments/analysis.ipynb) | [`experiments/passive_impact/custom_experiment/main.py`](experiments/passive_impact/custom_experiment/main.py) |
| **Aggressive Impact** | Market-order impact under propagator and hybrid models | [`experiments/agressive_impact/load_experiments/analysis.ipynb`](experiments/agressive_impact/load_experiments/analysis.ipynb) | [`experiments/agressive_impact/custom_experiment/main.py`](experiments/agressive_impact/custom_experiment/main.py) |
| **Queue Simulation** | Counterfactual queue paths under a metaorder (no impact curve) | [`experiments/queue_simulation/load_experiments/analysis.ipynb`](experiments/queue_simulation/load_experiments/analysis.ipynb) | [`experiments/queue_simulation/custom_experiment/main.py`](experiments/queue_simulation/custom_experiment/main.py) |

Each `load_experiments/` folder contains the notebook, plot utilities, and a `data/` subtree where the Rust binaries write `.npy` files. `.npy` is gitignored — regenerate baselines by running the binaries below. Each `custom_experiment/` folder contains a single `main.py` whose top section is a config dataclass; edit and run.

### Run a custom experiment (Python — primary user path)

Edit the config block at the top of the relevant `main.py`, then run it:

```bash
python experiments/passive_impact/custom_experiment/main.py
python experiments/agressive_impact/custom_experiment/main.py
python experiments/queue_simulation/custom_experiment/main.py
```

Outputs (`.npy` arrays) land in each folder's `output/` (gitignored). The Python facades produce the same shapes as the Rust binaries — see each `<Category>Config` dataclass in [`code/python/simproj/`](code/python/simproj/) for the full set of knobs (Hawkes parameters, queue parameters, metaorder shape, propagator vs. hybrid model, etc.).

### Inspect pre-saved baselines (notebooks)

Each category's `load_experiments/analysis.ipynb` loads `.npy` data from its sibling `data/` directory and renders the standard plots. The `.npy` files are gitignored — regenerate them via the Rust binaries below before opening the notebook on a fresh checkout.

### Regenerate baselines (Rust binaries)

The original Rust binaries still exist for fast batch baseline generation:

    cargo run --release --bin single_queue_efficient_with_us
    cargo run --release --bin single_queue_efficient_without_us
    cargo run --release --bin double_queue_efficient_with_us
    cargo run --release --bin double_queue_efficient_without_us
    cargo run --release --bin agressive_impact
    cargo run --release --bin agressive_impact_hybrid
    cargo run --release --bin queue_simulation_efficient

(General variants are also kept for validation: `*_general_with_us`, `*_general_without_us`.)

---

## Dependencies

- `numpy`, `scipy`: Numerical computing
- `numba`: JIT compilation for performance-critical Python code
- `maturin`, `pyo3`: Rust-to-Python bindings (development only)
- `pytest`: Testing framework
