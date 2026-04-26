# Experiments Restructure — Design

**Date:** 2026-04-26
**Branch:** `experiments_reworking`
**Scope:** `simulation-project/`

## Goal

Reorganise the `simulation-project/` repo so that:

1. Rust source and a new Python-bindings package live side by side under a `code/` parent.
2. Experiments are top-level (not nested under `python/`), grouped into three categories — `agressive_impact`, `passive_impact`, `queue_simulation` — each with the same internal shape: a `load_experiments/` folder for the pre-saved baseline (notebook + plot utilities + committed `.npy` data) and a `custom_experiment/` folder driven by a config-block `main.py` that calls the Python bindings.

The single-vs-double queue distinction becomes a config knob inside `passive_impact/` and `queue_simulation/`. The propagator-vs-hybrid distinction becomes a config knob inside `agressive_impact/`. `queue_simulation` is the counterfactual-queue subset of the passive-impact pipeline (no impact curve).

## Non-goals

- No deeper Rust deduplication of single/double queue code paths (see Section 4 below — option α only).
- No new analyses inside notebooks. Plots reproduce what they show today.
- No CI changes.
- No publication of the new Python package — local `maturin develop` only.

## Section 1 — Top-level directory layout

```
simulation-project/
├── code/
│   ├── src/                       # current src/ moved verbatim
│   │   ├── lib.rs
│   │   ├── models/ simulation/ ...
│   │   └── bin/                   # binaries kept; output paths repointed
│   ├── python/                    # maturin project (new)
│   │   ├── Cargo.toml             # PyO3 crate; depends on parent simulation_project lib
│   │   ├── pyproject.toml         # maturin build config
│   │   ├── src/lib.rs             # PyO3 wrappers for primitives
│   │   └── simproj/               # python package (facades + helpers)
│   └── Cargo.toml                 # workspace: members = ["src", "python"]
├── experiments/                   # new top-level (replaces python/experiments/)
│   ├── agressive_impact/
│   ├── passive_impact/
│   └── queue_simulation/
├── Cargo.toml                     # thin pointer to code/Cargo.toml workspace
├── README.md
└── .gitignore                     # adds custom_experiment/output/, _native*.so, etc.
```

Rationale and key choices:

- The crate name in `code/src/Cargo.toml` stays `simulation_project`, so existing `use simulation_project::...` paths in `code/src/bin/*` remain valid after the move.
- A Cargo workspace at `code/Cargo.toml` lets the `code/python/` PyO3 crate depend on the simulation library as a path dependency.
- A thin top-level `simulation-project/Cargo.toml` re-points to the workspace so `cargo run --release --bin <name>` from the repo root still works.
- The legacy top-level `data/` directory is removed; data is owned by each category's `load_experiments/` subtree.
- The legacy top-level `python/experiments/` directory is removed; its contents migrate per Section 5.

## Section 2 — Bindings package shape

```
code/python/
├── Cargo.toml          # crate "simproj_native", [lib] crate-type = ["cdylib"]
├── pyproject.toml      # maturin, module-name = "simproj._native"
├── src/lib.rs          # PyO3 wrappers (primitives only)
└── simproj/            # pure-Python package
    ├── __init__.py     # re-exports primitives + facades
    ├── _native.so      # produced by maturin develop (gitignored)
    ├── passive_impact.py
    ├── agressive_impact.py
    └── queue_simulation.py
```

### Approach (chosen): A — single maturin package, Rust primitives + Python facades

Discarded alternatives:

- **B (everything in Rust)**: facade tweaks would require `maturin develop` on every change. The facades are short orchestration code that gets edited often; keeping them in Python is the cheaper dev loop.
- **C (two-package layout, separate `_native` and `simproj` packages)**: only justified if `simproj` were published independently, which it is not.

### Primitives exposed by `_native` (PyO3, ~1:1 with Rust)

- `MultiExponentialHawkes`
- `AffineQueueProcess`
- `simulate(process, time_horizon, seed) -> SimulationResult`
- `simulate_with_externals(process, time_horizon, externals, seed) -> SimulationResult`
- `ConditionalSimulationContext`
- `TailImpact`
- `AggressiveImpactPath`
- Helpers: `hawkes_to_market_orders`, `merge_events`, `create_meta_orders`, `events_to_dim`, `extract_events_by_dim`, `sample_queue_at_times`

These accept/return numpy arrays where natural (event times, queue paths) and `#[pyclass]` wrappers otherwise (e.g. `Event`, `SimulationResult`).

### Facades (Python, in `simproj/<category>.py`)

Each module exposes:

- A `<Category>Config` dataclass with all knobs and sensible defaults.
- `run(config) -> dict[str, np.ndarray]` returning the same arrays the legacy binary writes today (e.g. `times`, `impact_paths`, `queue_paths`, `event_types`).
- `save(result, dir)` to persist those arrays as `.npy`.

The `metaorder` config field accepts either an `int` (count, evenly spaced inside `metaorder_window`) or an explicit `np.ndarray` of times — that is how custom metaorder shapes flow in.

### Module name

`simproj` — short, unambiguous in this project.

## Section 3 — Per-category folder pattern

```
experiments/<category>/
├── load_experiments/
│   ├── data/                       # .npy files committed to repo (baseline)
│   │   └── <variant subdirs>       # see Section 5
│   ├── plot_utils.py               # migrated from python/experiments/<x>/plot_utils.py
│   ├── images/                     # generated PNGs (committed)
│   └── analysis.ipynb              # existing notebook, paths repointed to ./data/
└── custom_experiment/
    ├── main.py                     # config block + facade call + plotting
    ├── output/                     # gitignored; fresh runs land here
    └── README.md                   # how to tweak config, where output goes
```

### `custom_experiment/main.py` skeleton (same shape across all three categories)

```python
from simproj import passive_impact as pi   # or agressive_impact / queue_simulation
from simproj.plotting import plot_all
import numpy as np

# ──────────────── CONFIG ────────────────
config = pi.PassiveImpactConfig(
    time_horizon=100.0,
    n_simulations=500,
    initial_queue_size=200,
    mode="single",                  # "single" | "double"
    side="both",                    # "with" | "without" | "both"
    # Hawkes
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    # Affine queue
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    # Metaorder: int → evenly spaced; np.ndarray → explicit times
    metaorder=375,
    metaorder_window=(1.0, 80.0),
    seed=42,
)
# ────────────────────────────────────────

result = pi.run(config)             # dict[str, np.ndarray]
pi.save(result, "output/")
plot_all(result, save_dir="output/images/")
```

`main.py` auto-generates plots after the run (last two lines) so the user gets a full result loop from one command.

### `load_experiments/analysis.ipynb`

The notebook content is unchanged. Only the data path constant in the imported `plot_utils.py` flips from `'../../../data/<x>'` to `'./data/'`.

### Plot utilities

Each category's `plot_utils.py` is the existing one, moved and rewired. `simproj.plotting.plot_all` is a tiny shim that dispatches to the right `plot_utils` based on the result's category (or each facade exposes its own `plot_all` wrapper — settled in implementation, not a design decision).

## Section 4 — Single/double queue Rust unification

### Approach (chosen): α — minimal, no Rust refactor

Today's Rust tree has parallel single/double modules:

- `src/models/queues/{queue_processes.rs, multiqueue_processes.rs}`
- `src/simulation_helpers/{single_queue, multi_queue}/`
- `src/conditional_impact/flow_imbalance_model/{single_queue, multi_queue}/`

Under α:

- The duplicated Rust paths stay as-is.
- Each Rust binary keeps its existing logic; only its **output path** is repointed (Section 5).
- The Python facade dispatches on `mode`: `mode="single"` calls into the single-queue primitives, `mode="double"` calls into the double-queue primitives.

Discarded alternative: **β — collapse the duplication** behind a generic `AffineQueueIntensity` parameterised by queue dimension. Cleaner long-term but a non-trivial Rust refactor that risks breaking conditional-impact and parity tests. Deferred to a separate follow-up branch.

## Section 5 — Migration of existing artifacts

### Code that moves verbatim (single `git mv`)

| Source | Destination |
|---|---|
| `src/` | `code/src/` |
| `Cargo.toml` | `code/Cargo.toml` (then add a thin root pointer + workspace member entries) |

### Notebooks + plot utilities

| Source | Destination |
|---|---|
| `python/experiments/single_queue_impact/` (notebook, plot_utils, images, README) | `experiments/passive_impact/load_experiments/` (canonical "single" view; double slots in alongside) |
| `python/experiments/double_queue_impact/` (notebook, plot_utils, images) | merged into `experiments/passive_impact/load_experiments/` — notebook gains a `mode = "single" \| "double"` cell at the top; `plot_utils.py` consolidates the two |
| `python/experiments/agressive_impact/` | `experiments/agressive_impact/load_experiments/` — `model = "propagator"` view |
| `python/experiments/agressive_impact_hybrid/` | merged into the same `load_experiments/` — `model = "hybrid"` view |
| `queue_simulation` notebook + plot_utils | **new** — derived from `single_queue_impact`'s plot_utils minus the impact panels |

### Pre-saved data

| Source | Destination |
|---|---|
| `data/single_queue/efficient/{with,without}/` | `experiments/passive_impact/load_experiments/data/single/efficient/{with,without}/` |
| `data/single_queue/general/{with,without}/` | `experiments/passive_impact/load_experiments/data/single/general/{with,without}/` |
| `data/double_queue/efficient/{with,without}/` | `experiments/passive_impact/load_experiments/data/double/efficient/{with,without}/` |
| `data/double_queue/general/{with,without}/` | `experiments/passive_impact/load_experiments/data/double/general/{with,without}/` |
| `data/agressive_impact/` | `experiments/agressive_impact/load_experiments/data/propagator/` |
| `data/agressive_impact_hybrid/` | `experiments/agressive_impact/load_experiments/data/hybrid/` |
| `queue_simulation` data | **new** — produced by a new lightweight binary (see below) |

### Rust binary output paths

Each binary's `let output_dir = "data/..."` constant is updated to point at the new tree. Examples:

- `code/src/bin/single_queue/efficient/with_us.rs` → writes to `experiments/passive_impact/load_experiments/data/single/efficient/with/`
- `code/src/bin/agressive_impact/main.rs` → writes to `experiments/agressive_impact/load_experiments/data/propagator/`
- `code/src/bin/agressive_impact_hybrid/main.rs` → writes to `experiments/agressive_impact/load_experiments/data/hybrid/`
- (and analogously for the `general` and `without_us` variants)

### New binary

`queue_simulation_efficient` — a stripped-down version of `single_queue_efficient_with_us` that produces `bar_q` paths but skips `TailImpact` setup and impact aggregation. Writes to `experiments/queue_simulation/load_experiments/data/{single,double}/efficient/`. Optionally a `_general` variant later.

### Orphan experiments (`cancelation_race/`, `extreme_events/`)

Moved as-is into `experiments/_legacy/` with no functional change. The `_legacy/` prefix marks them as not part of the new structure but preserves git history and visibility. They are not driven by the new Python facade and have no `custom_experiment/` counterpart.

## Section 6 — Build/run workflow & misc plumbing

### One-time setup

```bash
cd code/python && maturin develop --release && cd -
```

Builds the `_native` PyO3 module and installs `simproj` into the active Python environment (editable mode).

### Regenerate baseline data

```bash
cargo run --release --bin single_queue_efficient_with_us
cargo run --release --bin single_queue_efficient_without_us
cargo run --release --bin double_queue_efficient_with_us
cargo run --release --bin double_queue_efficient_without_us
cargo run --release --bin agressive_impact
cargo run --release --bin agressive_impact_hybrid
cargo run --release --bin queue_simulation_efficient   # new
# (and the *_general_* variants if desired)
```

Each writes into the appropriate `experiments/<cat>/load_experiments/data/<variant>/...` path.

### Run a custom experiment

```bash
python experiments/passive_impact/custom_experiment/main.py
# .npy + plots land in experiments/passive_impact/custom_experiment/output/
```

### Tests

- Rust tests under `code/src/` move with `git mv` and continue to run via `cargo test`.
- New: a single smoke test per category in `code/python/tests/` that calls the facade with a tiny `n_simulations=2, time_horizon=1.0` config and asserts the returned dict has the expected keys + array shapes. Catches binding regressions without paying full simulation cost.

### `.gitignore` additions

- `experiments/*/custom_experiment/output/`
- `code/python/target/`
- `code/python/simproj/_native*.so`
- `code/python/simproj.egg-info/`

### README

Replace the "Experiments" section in `README.md` with the new 3-category structure (load_experiments vs. custom_experiment), repoint embedded image paths.

## Validation criteria

This refactor is done when:

1. `cargo run --release --bin <any of the listed binaries>` from the repo root completes successfully and writes into `experiments/<cat>/load_experiments/data/<variant>/`.
2. `cd code/python && maturin develop --release` succeeds.
3. `python -c "from simproj import passive_impact, agressive_impact, queue_simulation; print('OK')"` succeeds.
4. `python experiments/passive_impact/custom_experiment/main.py` runs end-to-end with the default config and produces `.npy` + plots in `output/`.
5. Each `experiments/<cat>/load_experiments/analysis.ipynb` opens, runs all cells, and reproduces the existing baseline plots from the committed `.npy` data.
6. `cargo test` passes.
7. `pytest code/python/tests/` passes.

## Open implementation details (deferred to writing-plans)

- Exact PyO3 wrapper signatures for each primitive (numpy interop, error types).
- Whether `plot_all` lives in `simproj/plotting.py` or per-facade.
- Whether the `_general_*` variants get `custom_experiment` exposure or stay binary-only.
- Step ordering (workspace creation vs. `git mv` vs. data migration) — handled in the plan.
