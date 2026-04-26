# Experiments Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganise `simulation-project/` so Rust source and a new Python-bindings package live under `code/`, and experiments live at top-level under three categories (`agressive_impact`, `passive_impact`, `queue_simulation`), each with a `load_experiments/` baseline folder and a `custom_experiment/` config-driven `main.py`.

**Architecture:** `code/src/` keeps the Rust crate (renamed location, same crate name `simulation_project`). `code/python/` is a new maturin project whose PyO3 module `simproj._native` exposes the Rust primitives; pure-Python `simproj/<category>.py` facades wrap them with config dataclasses and a `run(config) -> dict[str, np.ndarray]` entry point. Existing Rust binaries are repointed to write into `experiments/<category>/load_experiments/data/<variant>/`. A new `queue_simulation_efficient` binary covers the queue-only category. Single-vs-double queue and propagator-vs-hybrid become Python facade dispatch knobs; no Rust deduplication.

**Tech Stack:** Rust 2021 (existing crate), PyO3 0.21+ via maturin, Python 3.10+, numpy, pandas (already used by notebooks), pytest.

**Spec reference:** [`docs/superpowers/specs/2026-04-26-experiments-restructure-design.md`](../specs/2026-04-26-experiments-restructure-design.md)

---

## Task 1: Move `src/` to `code/src/` and set up Cargo workspace

**Files:**
- Move: `src/` → `code/src/`
- Move: `Cargo.toml` → `code/Cargo.toml`
- Create: `Cargo.toml` (root, thin workspace pointer)
- Create: `code/Cargo.toml` (workspace root, replaces moved file's role)
- Create: `code/src/Cargo.toml` (renamed from old root Cargo.toml, becomes the lib package manifest)

The trick: the old root `Cargo.toml` contained both the `[package]` section for the `simulation_project` lib AND the `[[bin]]` entries pointing into `src/bin/`. After this task it becomes the lib-package manifest under `code/src/Cargo.toml`, and a new workspace manifest at `code/Cargo.toml` declares members.

- [ ] **Step 1: Move `src/` to `code/src/` with git mv**

```bash
mkdir code
git mv src code/src
```

- [ ] **Step 2: Move and rename the old root Cargo.toml under the lib package**

```bash
git mv Cargo.toml code/src/Cargo.toml
```

- [ ] **Step 3: Update binary paths inside `code/src/Cargo.toml`**

The binary paths in the moved file currently say `src/bin/...` but Cargo resolves paths relative to the manifest. Since the manifest now lives at `code/src/Cargo.toml` and binaries are at `code/src/bin/`, paths must become `bin/...`. Replace every `path = "src/bin/...` with `path = "bin/...`.

```bash
sed -i '' 's|path = "src/bin/|path = "bin/|g' code/src/Cargo.toml
```

(Verify by grepping: `grep 'path = ' code/src/Cargo.toml` should show no `src/bin/` prefixes.)

- [ ] **Step 4: Create the workspace manifest `code/Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = [
    "src",
    "python",
]

[workspace.package]
edition = "2021"
authors = ["Joseph Leclere"]
license = "MIT"

[profile.release]
opt-level = 3
lto = true
```

(The `python` member doesn't exist yet; that's OK — Task 4 creates it. Cargo will warn but won't fail until you build.)

- [ ] **Step 5: Create the thin root `Cargo.toml` pointer**

```toml
# Top-level pointer so `cargo` from repo root resolves to the workspace.
[workspace]
resolver = "2"
members = ["code/src"]
exclude = ["code/python"]

[profile.release]
opt-level = 3
lto = true
```

(Why exclude `code/python`: the python crate has its own workspace inside `code/`. Top-level workspace only owns the lib so root-level `cargo build --bin <x>` works.)

Wait — that creates two workspaces, which Cargo disallows. Resolve by making the **root** `Cargo.toml` the only workspace and removing `code/Cargo.toml`. Update Step 4 accordingly:

Replace Step 4's content with: skip creating `code/Cargo.toml`. Instead the root `Cargo.toml` is the workspace:

```toml
[workspace]
resolver = "2"
members = [
    "code/src",
    "code/python",   # added in Task 4
]

[profile.release]
opt-level = 3
lto = true
```

(Comment out or remove the `code/python` line until Task 4 lands. Alternatively leave it — Cargo errors clearly when a member dir is missing, and Task 4 is the next change.)

- [ ] **Step 6: Verify the workspace builds and binaries still work**

```bash
cargo build --release 2>&1 | tail -20
```
Expected: builds cleanly with no errors. Warnings about unused profile keys in `code/src/Cargo.toml` are OK (they're inherited from workspace) — silence them by removing the `[profile.release]` from `code/src/Cargo.toml`.

```bash
cargo run --release --bin agressive_impact 2>&1 | head -5
```
Expected: starts running normally and writes to `data/agressive_impact/` (output paths haven't been repointed yet — that's a later task; for now we're just verifying nothing broke).

(Cancel after a few seconds with Ctrl-C; actual data regen happens in Task 7.)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Move src/ under code/ and set up Cargo workspace"
```

---

## Task 2: Move existing pre-saved data into the new `experiments/` tree

**Files:**
- Move: `data/single_queue/` → `experiments/passive_impact/load_experiments/data/single/`
- Move: `data/double_queue/` → `experiments/passive_impact/load_experiments/data/double/`
- Move: `data/agressive_impact/` → `experiments/agressive_impact/load_experiments/data/propagator/`
- Move: `data/agressive_impact_hybrid/` → `experiments/agressive_impact/load_experiments/data/hybrid/`

Directory creation precedes `git mv`. After this task, the legacy `data/` directory is empty and removed.

- [ ] **Step 1: Create the experiments tree skeleton**

```bash
mkdir -p experiments/passive_impact/load_experiments/data/single
mkdir -p experiments/passive_impact/load_experiments/data/double
mkdir -p experiments/passive_impact/custom_experiment
mkdir -p experiments/agressive_impact/load_experiments/data
mkdir -p experiments/agressive_impact/custom_experiment
mkdir -p experiments/queue_simulation/load_experiments/data
mkdir -p experiments/queue_simulation/custom_experiment
mkdir -p experiments/_legacy
```

- [ ] **Step 2: Move passive_impact data**

```bash
git mv data/single_queue/efficient experiments/passive_impact/load_experiments/data/single/efficient
git mv data/single_queue/general experiments/passive_impact/load_experiments/data/single/general
git mv data/double_queue/efficient experiments/passive_impact/load_experiments/data/double/efficient
git mv data/double_queue/general experiments/passive_impact/load_experiments/data/double/general
```

- [ ] **Step 3: Move aggressive_impact data**

```bash
git mv data/agressive_impact experiments/agressive_impact/load_experiments/data/propagator
git mv data/agressive_impact_hybrid experiments/agressive_impact/load_experiments/data/hybrid
```

- [ ] **Step 4: Verify the legacy `data/` directory is empty and remove it**

```bash
find data -type f
rmdir data/single_queue data/double_queue data/experiments data 2>/dev/null || true
ls data 2>&1 || echo "data/ removed"
```

If `data/experiments/` or other subdirs still exist with files, inspect them — they may be stale outputs. If genuinely orphaned, `git rm -r data/<subdir>`.

- [ ] **Step 5: Verify .npy files are now under experiments/**

```bash
find experiments -name "*.npy" | head
```
Expected: lists files under `experiments/passive_impact/load_experiments/data/{single,double}/...` and `experiments/agressive_impact/load_experiments/data/{propagator,hybrid}/`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Move pre-saved baseline data into experiments/ tree"
```

---

## Task 3: Repoint Rust binary output paths to the new data tree

**Files to modify:**
- `code/src/bin/single_queue/efficient/with_us.rs` — line containing `"data/single_queue/efficient/with"`
- `code/src/bin/single_queue/efficient/without_us.rs` — `"data/single_queue/efficient/without"`
- `code/src/bin/single_queue/general/with_us.rs` — `"data/single_queue/general/with"`
- `code/src/bin/single_queue/general/without_us.rs` — `"data/single_queue/general/without"`
- `code/src/bin/double_queue/efficient/with_us.rs` — `"data/double_queue/efficient/with"`
- `code/src/bin/double_queue/efficient/without_us.rs` — `"data/double_queue/efficient/without"`
- `code/src/bin/double_queue/general/with_us.rs` — `"data/double_queue/general/with"`
- `code/src/bin/double_queue/general/without_us.rs` — `"data/double_queue/general/without"`
- `code/src/bin/agressive_impact/main.rs` — `"data/agressive_impact"`
- `code/src/bin/agressive_impact_hybrid/main.rs` — `"data/agressive_impact_hybrid"`

The mapping:

| Old path | New path |
|---|---|
| `data/single_queue/efficient/with` | `experiments/passive_impact/load_experiments/data/single/efficient/with` |
| `data/single_queue/efficient/without` | `experiments/passive_impact/load_experiments/data/single/efficient/without` |
| `data/single_queue/general/with` | `experiments/passive_impact/load_experiments/data/single/general/with` |
| `data/single_queue/general/without` | `experiments/passive_impact/load_experiments/data/single/general/without` |
| `data/double_queue/efficient/with` | `experiments/passive_impact/load_experiments/data/double/efficient/with` |
| `data/double_queue/efficient/without` | `experiments/passive_impact/load_experiments/data/double/efficient/without` |
| `data/double_queue/general/with` | `experiments/passive_impact/load_experiments/data/double/general/with` |
| `data/double_queue/general/without` | `experiments/passive_impact/load_experiments/data/double/general/without` |
| `data/agressive_impact` | `experiments/agressive_impact/load_experiments/data/propagator` |
| `data/agressive_impact_hybrid` | `experiments/agressive_impact/load_experiments/data/hybrid` |

- [ ] **Step 1: Apply the repointing for single_queue and double_queue binaries**

Edit each file listed above and replace the literal output-dir string. Each file has exactly one occurrence (the `let output_dir = "..."` near the bottom of `main`). Example for `code/src/bin/single_queue/efficient/with_us.rs`:

Search for: `"data/single_queue/efficient/with"`
Replace with: `"experiments/passive_impact/load_experiments/data/single/efficient/with"`

(Be careful: there is no trailing slash in the existing strings, and the binary's `std::fs::create_dir_all` call relies on this exact form. Match the existing form.)

Tip: a single sed pass works for these:
```bash
sed -i '' 's|"data/single_queue/|"experiments/passive_impact/load_experiments/data/single/|g' code/src/bin/single_queue/*/*.rs
sed -i '' 's|"data/double_queue/|"experiments/passive_impact/load_experiments/data/double/|g' code/src/bin/double_queue/*/*.rs
```

- [ ] **Step 2: Apply the repointing for the two aggressive_impact binaries**

```bash
sed -i '' 's|"data/agressive_impact"|"experiments/agressive_impact/load_experiments/data/propagator"|g' code/src/bin/agressive_impact/main.rs
sed -i '' 's|"data/agressive_impact_hybrid"|"experiments/agressive_impact/load_experiments/data/hybrid"|g' code/src/bin/agressive_impact_hybrid/main.rs
```

- [ ] **Step 3: Verify no `"data/` prefixes remain in any binary**

```bash
grep -rn '"data/' code/src/bin/
```
Expected: no matches.

- [ ] **Step 4: Build all binaries to ensure no syntax errors**

```bash
cargo build --release 2>&1 | tail -10
```
Expected: clean build.

- [ ] **Step 5: Smoke-run one binary and verify it writes to the new path**

```bash
cargo run --release --bin agressive_impact 2>&1 | tail -5
ls experiments/agressive_impact/load_experiments/data/propagator/*.npy
```
Expected: binary completes; `.npy` files appear at the new path. The files match what was migrated in Task 2 byte-for-byte (same default seeds and config).

- [ ] **Step 6: Verify byte-for-byte equality with migrated data**

```bash
# (assuming git tracks the migrated files, just-run output replaces them in working tree)
git diff --stat experiments/agressive_impact/load_experiments/data/propagator/
```
Expected: no diff (config in main.rs is unchanged, seed=42 fixed, deterministic).

If a small diff appears, investigate — but for `agressive_impact` with seed 42 it should be exact.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Repoint Rust binary outputs to new experiments/ data tree"
```

---

## Task 4: Create the maturin Python bindings crate skeleton

**Files:**
- Create: `code/python/Cargo.toml`
- Create: `code/python/pyproject.toml`
- Create: `code/python/src/lib.rs`
- Create: `code/python/simproj/__init__.py`
- Create: `code/python/README.md`
- Modify: `Cargo.toml` (root) — add `code/python` to workspace members
- Modify: `.gitignore`

After this task: `cd code/python && maturin develop --release` succeeds and `python -c "import simproj; print(simproj.__version__)"` works.

- [ ] **Step 1: Create `code/python/Cargo.toml`**

```toml
[package]
name = "simproj_native"
version = "0.1.0"
edition = "2021"
authors = ["Joseph Leclere"]
license = "MIT"

[lib]
name = "simproj_native"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
numpy = "0.21"
simulation_project = { path = "../src" }
```

- [ ] **Step 2: Create `code/python/pyproject.toml`**

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "simproj"
version = "0.1.0"
description = "Python bindings + facades for simulation_project"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "matplotlib>=3.7",
]

[tool.maturin]
module-name = "simproj._native"
python-source = "."
features = ["pyo3/extension-module"]
```

(The `python-source = "."` tells maturin the pure-Python files live at the same level as `pyproject.toml` — namely under `simproj/`. The native extension `_native.so` is dropped inside `simproj/` at install time.)

- [ ] **Step 3: Create the empty `_native` PyO3 module**

`code/python/src/lib.rs`:

```rust
use pyo3::prelude::*;

#[pymodule]
fn _native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
```

- [ ] **Step 4: Create the Python package init**

`code/python/simproj/__init__.py`:

```python
"""Python facades + bindings for the simulation_project Rust library."""
from . import _native  # noqa: F401

__version__ = _native.__version__
```

- [ ] **Step 5: Add `code/python` to the root workspace**

Edit the root `Cargo.toml` and add `"code/python"` to the workspace `members` array. Final contents:

```toml
[workspace]
resolver = "2"
members = [
    "code/src",
    "code/python",
]

[profile.release]
opt-level = 3
lto = true
```

- [ ] **Step 6: Add gitignore entries for build artifacts**

Append to `.gitignore`:
```
# maturin / pyo3 build artifacts
code/python/target/
code/python/simproj/_native*.so
code/python/simproj/_native*.pyd
code/python/simproj.egg-info/

# custom experiment outputs
experiments/*/custom_experiment/output/
```

- [ ] **Step 7: Build the bindings with maturin and verify the import works**

```bash
cd code/python && maturin develop --release && cd -
python -c "import simproj; print(simproj.__version__)"
```
Expected: prints `0.1.0`. If `maturin` is not installed: `pip install maturin`.

- [ ] **Step 8: Add a smoke test stub**

Create `code/python/tests/__init__.py` (empty) and `code/python/tests/test_skeleton.py`:

```python
def test_import_and_version():
    import simproj
    assert simproj.__version__ == "0.1.0"
```

Run:
```bash
cd code/python && pytest tests/test_skeleton.py -v && cd -
```
Expected: 1 passed.

- [ ] **Step 9: Create `code/python/README.md`**

```markdown
# simproj — Python bindings for simulation_project

## Build

    maturin develop --release

Re-run after any change to `src/lib.rs` (the PyO3 wrappers).

## Layout

- `src/lib.rs` — PyO3 wrappers for Rust primitives (compiled into `simproj._native`)
- `simproj/__init__.py` — re-exports
- `simproj/passive_impact.py` — facade for the passive impact experiment
- `simproj/agressive_impact.py` — facade for the aggressive impact experiment
- `simproj/queue_simulation.py` — facade for the queue-only experiment
- `tests/` — pytest smoke tests
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "Scaffold maturin bindings crate code/python with empty _native module"
```

---

## Task 5: Bind `MultiExponentialHawkes` and `simulate*` in PyO3

**Files:**
- Modify: `code/python/src/lib.rs`
- Modify: `code/python/simproj/__init__.py`
- Create: `code/python/tests/test_hawkes.py`

This task wraps the simplest primitive end-to-end so the wiring pattern is established.

- [ ] **Step 1: Replace `code/python/src/lib.rs` with the Hawkes binding**

```rust
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use simulation_project::models::MultiExponentialHawkes;
use simulation_project::simulation::simulate as rs_simulate;

#[pyclass(name = "MultiExponentialHawkes")]
#[derive(Clone)]
pub struct PyMultiExponentialHawkes {
    pub inner: MultiExponentialHawkes,
}

#[pymethods]
impl PyMultiExponentialHawkes {
    #[new]
    fn new(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        Self { inner: MultiExponentialHawkes::new(mu, alpha, beta) }
    }

    #[staticmethod]
    fn with_stationary_state(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        let base = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
        let inner = MultiExponentialHawkes::new_with_state(
            base.stationary_state(), mu, alpha, beta,
        );
        Self { inner }
    }

    fn stationary_state<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.inner.stationary_state().into_pyarray(py)
    }

    fn m(&self) -> usize { self.inner.m() }
}

/// Simulate the Hawkes process up to `t_max`. Returns the event times as
/// a 1-D numpy array of f64 (dim info is dropped — Hawkes is single-dim).
#[pyfunction]
fn simulate_hawkes<'py>(
    py: Python<'py>,
    hawkes: &PyMultiExponentialHawkes,
    t_max: f64,
    seed: Option<u64>,
) -> &'py PyArray1<f64> {
    let result = rs_simulate(&hawkes.inner, t_max, seed);
    let times: Vec<f64> = result.events.iter().map(|e| e.time).collect();
    times.into_pyarray(py)
}

#[pymodule]
fn _native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyMultiExponentialHawkes>()?;
    m.add_function(wrap_pyfunction!(simulate_hawkes, m)?)?;
    Ok(())
}
```

- [ ] **Step 2: Re-export from `simproj/__init__.py`**

```python
"""Python facades + bindings for the simulation_project Rust library."""
from . import _native

MultiExponentialHawkes = _native.MultiExponentialHawkes
simulate_hawkes = _native.simulate_hawkes

__version__ = _native.__version__
```

- [ ] **Step 3: Rebuild and write the smoke test**

```bash
cd code/python && maturin develop --release && cd -
```

Create `code/python/tests/test_hawkes.py`:

```python
import numpy as np
import simproj


def test_hawkes_stationary_state():
    h = simproj.MultiExponentialHawkes.with_stationary_state(
        mu=1.0, alpha=[0.065, 0.2], beta=[0.15, 0.6],
    )
    state = h.stationary_state()
    assert state.shape == (2,)
    assert np.all(state > 0)


def test_hawkes_simulate_returns_event_times():
    h = simproj.MultiExponentialHawkes(
        mu=1.0, alpha=[0.065, 0.2, 0.325, 0.65], beta=[0.15, 0.6, 2.5, 10.0],
    )
    times = simproj.simulate_hawkes(h, t_max=10.0, seed=42)
    assert times.dtype == np.float64
    assert times.ndim == 1
    assert len(times) > 0
    assert np.all(np.diff(times) >= 0)  # strictly increasing
```

- [ ] **Step 4: Run the test**

```bash
cd code/python && pytest tests/test_hawkes.py -v && cd -
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Bind MultiExponentialHawkes and simulate_hawkes in PyO3"
```

---

## Task 6: Bind `AffineQueueProcess` (single + double) and externals helpers

**Files:**
- Modify: `code/python/src/lib.rs`
- Modify: `code/python/simproj/__init__.py`
- Create: `code/python/tests/test_queue.py`

Goal: expose the full minimal pipeline needed by `passive_impact` / `queue_simulation` facades — affine queue creation + `simulate_with_externals` + the helpers `hawkes_to_market_orders`, `merge_events`, `create_meta_orders`, `extract_events_by_dim`, `sample_queue_at_times`. Done in one task because they're tightly coupled (each helper takes/returns the same `MultivariateSimulationResult`).

A thin `PySimulationResult` wraps `MultivariateSimulationResult` (which is a struct of `Vec<MultivariateEvent>`) so it can be passed between PyO3 functions opaquely. Internal accessors return numpy arrays.

- [ ] **Step 1: Append the simulation result wrapper and queue bindings to `code/python/src/lib.rs`**

Add the imports at the top (extend the existing `use` block):

```rust
use simulation_project::models::{
    AffineQueueProcess, AffineBidAskQueueProcess, BidAskAffineParams, AffineIntensityParams,
    MarkovianProcess, MultivariateSimulationResult,
};
use simulation_project::simulation::simulate_with_externals as rs_simulate_with_externals;
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders as rs_hawkes_to_market_orders,
    merge_events as rs_merge_events,
    create_meta_orders as rs_create_meta_orders,
    events_to_dim as rs_events_to_dim,
    extract_events_by_dim as rs_extract_events_by_dim,
    sample_queue_at_times as rs_sample_queue_at_times,
};
use numpy::{PyArray2, PyReadonlyArray1};
```

Then append the new pyclasses + functions:

```rust
/// Opaque wrapper for a MultivariateSimulationResult (a list of
/// (time, dim) event tuples). Pass between functions; convert to
/// numpy via .times() / .dims() when needed.
#[pyclass(name = "SimulationResult")]
#[derive(Clone)]
pub struct PySimulationResult {
    pub inner: MultivariateSimulationResult,
}

#[pymethods]
impl PySimulationResult {
    fn times<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        let v: Vec<f64> = self.inner.events.iter().map(|e| e.time).collect();
        v.into_pyarray(py)
    }
    fn dims<'py>(&self, py: Python<'py>) -> &'py PyArray1<usize> {
        let v: Vec<usize> = self.inner.events.iter().map(|e| e.dim).collect();
        v.into_pyarray(py)
    }
    fn __len__(&self) -> usize { self.inner.events.len() }
}

/// Opaque wrapper for a MarkovianProcess (the queue process produced
/// by AffineQueueProcess::new_queue / new). Treated as black-box.
#[pyclass(name = "QueueProcess")]
pub struct PyQueueProcess {
    pub inner: MarkovianProcess,
}

#[pymethods]
impl PyQueueProcess {
    fn dim(&self) -> usize {
        // MarkovianProcess::dim returns usize via MultivariateMarkovianIntensity trait
        use simulation_project::models::MultivariateMarkovianIntensity;
        self.inner.dim()
    }
}

#[pyclass(name = "AffineQueueProcess")]
pub struct PyAffineQueueProcess;

#[pymethods]
impl PyAffineQueueProcess {
    /// Decoupled single-queue process (state = [q]). Market orders dim=2 must
    /// be supplied as externals.
    #[staticmethod]
    fn new_queue(q0: f64, a_l: f64, b_l: f64, a_c: f64, b_c: f64) -> PyQueueProcess {
        PyQueueProcess { inner: AffineQueueProcess::new_queue(q0, a_l, b_l, a_c, b_c) }
    }

    /// Static helper: c_lambda = b_c - b_l.
    #[staticmethod]
    fn c_lambda(b_l: f64, b_c: f64) -> f64 { AffineQueueProcess::c_lambda(b_l, b_c) }
}

#[pyclass(name = "AffineBidAskQueueProcess")]
pub struct PyAffineBidAskQueueProcess;

#[pymethods]
impl PyAffineBidAskQueueProcess {
    /// Decoupled double-queue process (state = [q_a, q_b]).
    /// Limit/cancel intensities are affine in (q_a, q_b) per side.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn new_queue(
        q0_a: f64, q0_b: f64,
        // ask side: λ^L_a = a + b_aa*q_a + b_ab*q_b, similarly for cancel
        l_a_const: f64, l_a_self: f64, l_a_cross: f64,
        c_a_const: f64, c_a_self: f64, c_a_cross: f64,
        // bid side
        l_b_const: f64, l_b_self: f64, l_b_cross: f64,
        c_b_const: f64, c_b_self: f64, c_b_cross: f64,
    ) -> PyQueueProcess {
        // Note: BidAskAffineParams::new takes 4 AffineIntensityParams (l_a, c_a, l_b, c_b).
        // Each AffineIntensityParams stores (a, b_a, b_b) = constant + slope on q_a + slope on q_b.
        let params = BidAskAffineParams::new(
            AffineIntensityParams::new(l_a_const, l_a_self, l_a_cross),
            AffineIntensityParams::new(c_a_const, c_a_self, c_a_cross),
            AffineIntensityParams::new(l_b_const, l_b_self, l_b_cross),
            AffineIntensityParams::new(c_b_const, c_b_self, c_b_cross),
        );
        PyQueueProcess { inner: AffineBidAskQueueProcess::new_queue(q0_a, q0_b, params) }
    }
}

/// Simulate process with external events (e.g. market orders driven by Hawkes).
#[pyfunction]
fn simulate_with_externals(
    process: &PyQueueProcess,
    t_max: f64,
    externals: &PySimulationResult,
    seed: Option<u64>,
) -> PySimulationResult {
    PySimulationResult {
        inner: rs_simulate_with_externals(&process.inner, t_max, &externals.inner, seed),
    }
}

/// Simulate Hawkes and return as a SimulationResult marked dim=2 (market orders).
#[pyfunction]
fn simulate_hawkes_as_market_orders(
    hawkes: &PyMultiExponentialHawkes,
    t_max: f64,
    seed: Option<u64>,
) -> PySimulationResult {
    let result = rs_simulate(&hawkes.inner, t_max, seed);
    PySimulationResult { inner: rs_hawkes_to_market_orders(&result) }
}

#[pyfunction]
fn merge_events(a: &PySimulationResult, b: &PySimulationResult) -> PySimulationResult {
    PySimulationResult { inner: rs_merge_events(&a.inner, &b.inner) }
}

/// Build an evenly-spaced metaorder block of n orders from t_start to t_end,
/// tagged at dim=0 (overridable via events_to_dim).
#[pyfunction]
fn create_meta_orders(n: u32, t_start: f64, t_end: f64) -> PySimulationResult {
    PySimulationResult { inner: rs_create_meta_orders(n, t_start, t_end) }
}

/// Build a metaorder from an explicit list of times; tagged at dim=2 (market).
/// total_dims is required so downstream simulations know the dim count.
#[pyfunction]
fn create_meta_orders_from_times(
    times: PyReadonlyArray1<f64>,
    target_dim: usize,
    total_dims: usize,
) -> PySimulationResult {
    use simulation_project::models::{MultivariateEvent, MultivariateSimulationResult};
    let mut result = MultivariateSimulationResult::new(total_dims);
    for &t in times.as_slice().unwrap() {
        result.events.push(MultivariateEvent { time: t, dim: target_dim });
    }
    PySimulationResult { inner: result }
}

#[pyfunction]
fn events_to_dim(events: &PySimulationResult, target_dim: usize, total_dims: usize) -> PySimulationResult {
    PySimulationResult { inner: rs_events_to_dim(&events.inner, target_dim, total_dims) }
}

/// Returns a list of f64 vectors per dim. Cardinality = total_dims unless `exclude_dim` set.
#[pyfunction]
fn extract_events_by_dim<'py>(
    py: Python<'py>,
    result: &PySimulationResult,
    total_dims: usize,
    exclude_dim: Option<usize>,
) -> Vec<&'py PyArray1<f64>> {
    let by_dim = rs_extract_events_by_dim(&result.inner, total_dims, exclude_dim);
    by_dim.into_iter().map(|v| v.into_pyarray(py)).collect()
}

#[pyfunction]
fn sample_queue_at_times<'py>(
    py: Python<'py>,
    queue_path_events: &PySimulationResult,
    initial_q: u32,
    times: PyReadonlyArray1<f64>,
) -> &'py PyArray1<u32> {
    use simulation_project::models::AffineQueueProcess;
    let q_path = AffineQueueProcess::result_to_queue_path(&queue_path_events.inner, initial_q);
    let samples = rs_sample_queue_at_times(&q_path, times.as_slice().unwrap());
    samples.into_pyarray(py)
}
```

Then update the `#[pymodule]` registration block:

```rust
#[pymodule]
fn _native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyMultiExponentialHawkes>()?;
    m.add_class::<PySimulationResult>()?;
    m.add_class::<PyQueueProcess>()?;
    m.add_class::<PyAffineQueueProcess>()?;
    m.add_class::<PyAffineBidAskQueueProcess>()?;
    m.add_function(wrap_pyfunction!(simulate_hawkes, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_with_externals, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hawkes_as_market_orders, m)?)?;
    m.add_function(wrap_pyfunction!(merge_events, m)?)?;
    m.add_function(wrap_pyfunction!(create_meta_orders, m)?)?;
    m.add_function(wrap_pyfunction!(create_meta_orders_from_times, m)?)?;
    m.add_function(wrap_pyfunction!(events_to_dim, m)?)?;
    m.add_function(wrap_pyfunction!(extract_events_by_dim, m)?)?;
    m.add_function(wrap_pyfunction!(sample_queue_at_times, m)?)?;
    Ok(())
}
```

- [ ] **Step 2: Re-export new symbols in `simproj/__init__.py`**

```python
"""Python facades + bindings for the simulation_project Rust library."""
from . import _native

MultiExponentialHawkes = _native.MultiExponentialHawkes
SimulationResult = _native.SimulationResult
AffineQueueProcess = _native.AffineQueueProcess
AffineBidAskQueueProcess = _native.AffineBidAskQueueProcess

simulate_hawkes = _native.simulate_hawkes
simulate_with_externals = _native.simulate_with_externals
simulate_hawkes_as_market_orders = _native.simulate_hawkes_as_market_orders
merge_events = _native.merge_events
create_meta_orders = _native.create_meta_orders
create_meta_orders_from_times = _native.create_meta_orders_from_times
events_to_dim = _native.events_to_dim
extract_events_by_dim = _native.extract_events_by_dim
sample_queue_at_times = _native.sample_queue_at_times

__version__ = _native.__version__
```

- [ ] **Step 3: Rebuild and write smoke test**

```bash
cd code/python && maturin develop --release && cd -
```

Create `code/python/tests/test_queue.py`:

```python
import numpy as np
import simproj


def test_affine_queue_smoke():
    hawkes = simproj.MultiExponentialHawkes.with_stationary_state(
        mu=1.0, alpha=[0.065, 0.2, 0.325, 0.65], beta=[0.15, 0.6, 2.5, 10.0],
    )
    market_orders = simproj.simulate_hawkes_as_market_orders(hawkes, t_max=10.0, seed=42)
    assert len(market_orders) > 0

    process = simproj.AffineQueueProcess.new_queue(
        q0=200.0, a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    )
    q_events = simproj.simulate_with_externals(process, 10.0, market_orders, seed=42)
    full = simproj.merge_events(q_events, market_orders)
    samples = simproj.sample_queue_at_times(
        full, initial_q=200,
        times=np.linspace(0.0, 10.0, 11).astype(np.float64),
    )
    assert samples.shape == (11,)
    assert samples.dtype == np.uint32


def test_create_meta_orders_explicit_times():
    times = np.array([1.0, 2.0, 4.0, 8.0])
    meta = simproj.create_meta_orders_from_times(times, target_dim=2, total_dims=3)
    assert len(meta) == 4
    assert np.allclose(meta.times(), times)
    assert np.all(meta.dims() == 2)


def test_c_lambda_helper():
    assert simproj.AffineQueueProcess.c_lambda(b_l=-0.275, b_c=0.125) == 0.4
```

- [ ] **Step 4: Run tests**

```bash
cd code/python && pytest tests/test_queue.py -v && cd -
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Bind AffineQueueProcess (single + double) and externals helpers"
```

---

## Task 7: Bind `ConditionalSimulationContext`

**Files:**
- Modify: `code/python/src/lib.rs`
- Modify: `code/python/simproj/__init__.py`
- Create: `code/python/tests/test_conditional.py`

The `ConditionalSimulationContext` lives by reference (it borrows from the process and the conditioning events). Wrapping a borrowing API in PyO3 requires owning the inputs inside the wrapper. The cleanest approach: `PyConditionalSimulationContext` owns clones of all inputs needed and constructs a fresh Rust `ConditionalSimulationContext` on each `simulate_*` call.

- [ ] **Step 1: Append the conditional simulator binding to `code/python/src/lib.rs`**

Add imports:
```rust
use simulation_project::simulation::ConditionalSimulationContext;
```

Add the wrapper:

```rust
/// Conditional simulation context. Owns its inputs (conditioning events + externals + process)
/// and rebuilds the borrowed Rust context on each call.
#[pyclass(name = "ConditionalSimulationContext")]
pub struct PyConditionalSimulationContext {
    process: Py<PyQueueProcess>,
    cond_events_by_dim: Vec<Vec<f64>>,
    cond_externals: Option<MultivariateSimulationResult>,
    new_externals: Option<MultivariateSimulationResult>,
    t_max: f64,
}

#[pymethods]
impl PyConditionalSimulationContext {
    #[new]
    fn new(
        process: Py<PyQueueProcess>,
        cond_events_by_dim: Vec<Vec<f64>>,
        cond_externals: Option<&PySimulationResult>,
        new_externals: Option<&PySimulationResult>,
        t_max: f64,
    ) -> Self {
        Self {
            process,
            cond_events_by_dim,
            cond_externals: cond_externals.map(|r| r.inner.clone()),
            new_externals: new_externals.map(|r| r.inner.clone()),
            t_max,
        }
    }

    /// Memory-efficient queue sampling at specified times. Returns a 1-D numpy
    /// array of u32 queue values aligned with `times`.
    fn simulate_queue_at_times<'py>(
        &self,
        py: Python<'py>,
        times: PyReadonlyArray1<f64>,
        initial_queue_size: u32,
        seed: Option<u64>,
    ) -> &'py PyArray1<u32> {
        let process_borrow = self.process.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &process_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );
        let samples = ctx.simulate_queue_at_times(
            times.as_slice().unwrap(),
            initial_queue_size,
            None,        // new_initial_state — not currently exposed; spec uses default
            seed,
        );
        samples.into_pyarray(py)
    }

    /// Single-shot conditional simulate; returns the resulting event stream.
    fn simulate(
        &self,
        py: Python,
        seed: Option<u64>,
    ) -> PySimulationResult {
        let process_borrow = self.process.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &process_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );
        PySimulationResult { inner: ctx.simulate(None, seed) }
    }
}
```

Register in the `#[pymodule]` block:
```rust
m.add_class::<PyConditionalSimulationContext>()?;
```

- [ ] **Step 2: Re-export in `simproj/__init__.py`**

Add the line:
```python
ConditionalSimulationContext = _native.ConditionalSimulationContext
```

- [ ] **Step 3: Rebuild and add smoke test**

```bash
cd code/python && maturin develop --release && cd -
```

Create `code/python/tests/test_conditional.py`:

```python
import numpy as np
import simproj


def test_conditional_simulate_queue_at_times():
    hawkes = simproj.MultiExponentialHawkes.with_stationary_state(
        mu=1.0, alpha=[0.065, 0.2, 0.325, 0.65], beta=[0.15, 0.6, 2.5, 10.0],
    )
    market_orders = simproj.simulate_hawkes_as_market_orders(hawkes, t_max=10.0, seed=42)

    process = simproj.AffineQueueProcess.new_queue(
        q0=200.0, a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    )
    q_events = simproj.simulate_with_externals(process, 10.0, market_orders, seed=42)
    cond_by_dim = simproj.extract_events_by_dim(q_events, total_dims=3, exclude_dim=2)

    meta = simproj.create_meta_orders(n=10, t_start=1.0, t_end=8.0)
    bar_q_externals = simproj.merge_events(meta, market_orders)

    ctx = simproj.ConditionalSimulationContext(
        process,
        [list(arr) for arr in cond_by_dim],
        cond_externals=market_orders,
        new_externals=bar_q_externals,
        t_max=10.0,
    )

    times = np.linspace(0.0, 10.0, 11).astype(np.float64)
    samples = ctx.simulate_queue_at_times(times, initial_queue_size=200, seed=0)
    assert samples.shape == (11,)
    assert samples.dtype == np.uint32
```

- [ ] **Step 4: Run test**

```bash
cd code/python && pytest tests/test_conditional.py -v && cd -
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Bind ConditionalSimulationContext"
```

---

## Task 8: Bind `TailImpact` and `AggressiveImpactPath`

**Files:**
- Modify: `code/python/src/lib.rs`
- Modify: `code/python/simproj/__init__.py`
- Create: `code/python/tests/test_impact.py`

`AggressiveImpactPath::from_queue_samples` takes a `kappa: impl Fn(f64) -> f64`. To accept a Python callable, we capture it as `Py<PyAny>` and call back into Python with the GIL inside a Rust closure. This works for non-perf-critical evaluation paths (called once per event time, not per RNG draw).

- [ ] **Step 1: Append impact bindings to `code/python/src/lib.rs`**

Add imports:
```rust
use simulation_project::conditional_impact::{TailImpact, AggressiveImpactPath};
```

Add the wrappers:

```rust
#[pyclass(name = "TailImpact")]
pub struct PyTailImpact {
    pub inner: TailImpact,
}

#[pymethods]
impl PyTailImpact {
    /// Build TailImpact from affine-queue parameters.
    #[staticmethod]
    fn from_affine_queue(
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
        b_l: f64,
        b_c: f64,
        events: Vec<f64>,
    ) -> Self {
        Self { inner: TailImpact::from_affine_queue(mu, alpha, beta, b_l, b_c, events) }
    }
}

#[pyclass(name = "AggressiveImpactPath")]
pub struct PyAggressiveImpactPath {
    pub impact_path: Vec<f64>,
}

#[pymethods]
impl PyAggressiveImpactPath {
    fn impact<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.impact_path.clone().into_pyarray(py)
    }
}

/// Compute aggressive impact path from pre-sampled queues.
/// `kappa` is a Python callable f64 -> f64 invoked at each evaluation time.
#[pyfunction]
fn aggressive_impact_from_queue_samples(
    py: Python,
    q_samples: PyReadonlyArray1<u32>,
    q_bar_samples: PyReadonlyArray1<u32>,
    eval_times: PyReadonlyArray1<f64>,
    is_market_order: Vec<bool>,
    hawkes: &PyMultiExponentialHawkes,
    kappa: PyObject,
) -> PyResult<PyAggressiveImpactPath> {
    let kappa_clone = kappa.clone_ref(py);
    let path = AggressiveImpactPath::from_queue_samples(
        q_samples.as_slice().unwrap(),
        q_bar_samples.as_slice().unwrap(),
        eval_times.as_slice().unwrap(),
        &is_market_order,
        &hawkes.inner,
        |q: f64| -> f64 {
            Python::with_gil(|py| {
                let res = kappa_clone.call1(py, (q,)).unwrap();
                res.extract::<f64>(py).unwrap()
            })
        },
    );
    Ok(PyAggressiveImpactPath { impact_path: path.impact_path })
}
```

Register in the `#[pymodule]` block:
```rust
m.add_class::<PyTailImpact>()?;
m.add_class::<PyAggressiveImpactPath>()?;
m.add_function(wrap_pyfunction!(aggressive_impact_from_queue_samples, m)?)?;
```

- [ ] **Step 2: Re-export in `simproj/__init__.py`**

```python
TailImpact = _native.TailImpact
AggressiveImpactPath = _native.AggressiveImpactPath
aggressive_impact_from_queue_samples = _native.aggressive_impact_from_queue_samples
```

- [ ] **Step 3: Rebuild and write smoke test**

```bash
cd code/python && maturin develop --release && cd -
```

Create `code/python/tests/test_impact.py`:

```python
import numpy as np
import simproj


def test_tail_impact_from_affine():
    ti = simproj.TailImpact.from_affine_queue(
        mu=1.0, alpha=[0.065, 0.2], beta=[0.15, 0.6],
        b_l=-0.275, b_c=0.125, events=[1.0, 2.0, 3.0],
    )
    assert ti is not None  # opaque smoke


def test_aggressive_impact_from_samples():
    hawkes = simproj.MultiExponentialHawkes(
        mu=1.0, alpha=[0.065, 0.2], beta=[0.15, 0.6],
    )
    n = 10
    q = np.full(n, 200, dtype=np.uint32)
    q_bar = np.full(n, 180, dtype=np.uint32)
    times = np.linspace(0.0, 10.0, n).astype(np.float64)
    is_market = [True] * n

    def kappa(q):
        return 1000.0 * np.sqrt(np.log(np.exp(-0.01 * q) + 1.0))

    result = simproj.aggressive_impact_from_queue_samples(
        q_samples=q, q_bar_samples=q_bar,
        eval_times=times, is_market_order=is_market,
        hawkes=hawkes, kappa=kappa,
    )
    impact = result.impact()
    assert impact.shape == (n,)
    assert impact.dtype == np.float64
```

- [ ] **Step 4: Run test**

```bash
cd code/python && pytest tests/test_impact.py -v && cd -
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Bind TailImpact and AggressiveImpactPath (Python kappa callable)"
```

---

## Task 9: Migrate notebooks + plot utilities into `experiments/`

**Files:**
- Move: `python/experiments/single_queue_impact/{plot_utils.py,discovery.ipynb,images/,README.md}` → `experiments/passive_impact/load_experiments/`
- Move: `python/experiments/double_queue_impact/{plot_utils.py,analysis.ipynb,images/,README.md}` → temporary location for merge
- Move: `python/experiments/agressive_impact/{plot_utils.py,analysis.ipynb,images/,README.md}` → `experiments/agressive_impact/load_experiments/`
- Move: `python/experiments/agressive_impact_hybrid/{plot_utils.py,images/}` → temporary location for merge
- Move: `python/experiments/cancelation_race/`, `python/experiments/extreme_events/` → `experiments/_legacy/`
- Modify: copied `plot_utils.py` files to repoint `DATA_BASE` to `./data/...`
- Create: `experiments/passive_impact/load_experiments/plot_utils.py` (consolidated single+double)
- Create: `experiments/agressive_impact/load_experiments/plot_utils.py` (consolidated propagator+hybrid)

The merge step is small but real: each pair (single+double, propagator+hybrid) currently has two near-identical `plot_utils.py` files. The notebooks still need to render exactly the same plots they render today, so the merge has to preserve every plotting function from each side.

- [ ] **Step 1: Move passive_impact baseline notebook + plot_utils**

```bash
git mv python/experiments/single_queue_impact/discovery.ipynb experiments/passive_impact/load_experiments/analysis.ipynb
git mv python/experiments/single_queue_impact/plot_utils.py experiments/passive_impact/load_experiments/plot_utils_single.py
git mv python/experiments/single_queue_impact/images experiments/passive_impact/load_experiments/images
git mv python/experiments/single_queue_impact/README.md experiments/passive_impact/load_experiments/README.md
git mv python/experiments/single_queue_impact/impact_given_q.pdf experiments/passive_impact/load_experiments/impact_given_q.pdf 2>/dev/null || true
git mv python/experiments/single_queue_impact/impact_given_barq.pdf experiments/passive_impact/load_experiments/impact_given_barq.pdf 2>/dev/null || true

git mv python/experiments/double_queue_impact/plot_utils.py experiments/passive_impact/load_experiments/plot_utils_double.py
git mv python/experiments/double_queue_impact/analysis.ipynb experiments/passive_impact/load_experiments/analysis_double.ipynb
# Move double images alongside single, prefixed to avoid collisions
mkdir -p experiments/passive_impact/load_experiments/images_double
git mv python/experiments/double_queue_impact/images/* experiments/passive_impact/load_experiments/images_double/
git rm -r python/experiments/double_queue_impact/images
git mv python/experiments/double_queue_impact/README.md experiments/passive_impact/load_experiments/README_double.md 2>/dev/null || true
```

- [ ] **Step 2: Repoint DATA_BASE in `plot_utils_single.py`**

Open `experiments/passive_impact/load_experiments/plot_utils_single.py`. The current code reads:

```python
data_base = f'../../../data/single_queue/{data_mode}'
```

Replace with:

```python
data_base = f'./data/single/{data_mode}'
```

(Or find/replace any other path-construction lines in this file pointing to `data/`.)

- [ ] **Step 3: Repoint DATA_BASE in `plot_utils_double.py`**

In `experiments/passive_impact/load_experiments/plot_utils_double.py`, replace path constants pointing at `../../../data/double_queue/...` with `./data/double/...`.

- [ ] **Step 4: Create the consolidated `plot_utils.py`**

Combine both modules into one file. The skeleton:

```python
"""Plot utilities for passive impact experiments (single and double queue).

Loads pre-saved baseline data from ./data/{single,double}/{efficient,general}/{with,without}/.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Plot passive impact results.')
    parser.add_argument('--mode', choices=['single', 'double'], default='single')
    parser.add_argument('--data-mode', choices=['general', 'efficient'], default='efficient')
    parser.add_argument('--meta-end', type=float, default=80.0)
    return parser.parse_args()


# ─── Loaders ──────────────────────────────────────────────────────────

def load_single(data_mode):
    """Load single-queue results."""
    base = f'./data/single/{data_mode}'
    # ... (paste the body of load_data() from plot_utils_single.py, swap the
    # base path at the top to `base`)

def load_double(data_mode):
    """Load double-queue results."""
    base = f'./data/double/{data_mode}'
    # ... (paste from plot_utils_double.py)


# ─── Plotters ─────────────────────────────────────────────────────────

def plot_queue_shades(df, sim_col, title, label, meta_end, ref_col=None, save_path=None):
    # paste verbatim from plot_utils_single.py (it's the only plotting function used)
    ...


# ─── Top-level ────────────────────────────────────────────────────────

def generate_all_plots(mode, data_mode, meta_end):
    if mode == 'single':
        path_with, path_without, queue_with, queue_without = load_single(data_mode)
    else:
        # double-queue equivalent — uses the loader from plot_utils_double.py
        ...
    # Plot — same call structure as today's generate_all_plots.


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(args.mode, args.data_mode, args.meta_end)
```

Once consolidated and working, delete the two helper files:
```bash
git rm experiments/passive_impact/load_experiments/plot_utils_single.py
git rm experiments/passive_impact/load_experiments/plot_utils_double.py
```

(If consolidation is involved enough that you'd rather keep them as `_single.py` / `_double.py` modules and have the new `plot_utils.py` be a thin dispatcher, that's fine — the requirement is just one entry-point that handles both.)

- [ ] **Step 5: Move agressive_impact baseline notebook + plot_utils**

```bash
git mv python/experiments/agressive_impact/analysis.ipynb experiments/agressive_impact/load_experiments/analysis.ipynb
git mv python/experiments/agressive_impact/plot_utils.py experiments/agressive_impact/load_experiments/plot_utils_propagator.py
git mv python/experiments/agressive_impact/images experiments/agressive_impact/load_experiments/images
git mv python/experiments/agressive_impact/README.md experiments/agressive_impact/load_experiments/README.md

git mv python/experiments/agressive_impact_hybrid/plot_utils.py experiments/agressive_impact/load_experiments/plot_utils_hybrid.py
mkdir -p experiments/agressive_impact/load_experiments/images_hybrid
git mv python/experiments/agressive_impact_hybrid/images/* experiments/agressive_impact/load_experiments/images_hybrid/
git rm -r python/experiments/agressive_impact_hybrid/images
```

- [ ] **Step 6: Repoint DATA_BASE in propagator and hybrid plot_utils**

In `plot_utils_propagator.py`, replace the line `DATA_BASE = '../../../data/agressive_impact'` with `DATA_BASE = './data/propagator'`.

In `plot_utils_hybrid.py`, replace the equivalent constant (likely `DATA_BASE = '../../../data/agressive_impact_hybrid'`) with `DATA_BASE = './data/hybrid'`.

- [ ] **Step 7: Create consolidated `experiments/agressive_impact/load_experiments/plot_utils.py`**

Same pattern as Step 4 — one entry point with `--model {propagator,hybrid}` arg. The two source files have nearly identical plot functions; consolidate.

- [ ] **Step 8: Move orphan experiments to `_legacy/`**

```bash
git mv python/experiments/cancelation_race experiments/_legacy/cancelation_race
git mv python/experiments/extreme_events experiments/_legacy/extreme_events
```

- [ ] **Step 9: Remove the now-empty `python/experiments/` and `python/` directories**

```bash
find python -type f
# If only __pycache__ files remain:
rm -rf python/experiments/__pycache__ python/experiments/*/{__pycache__,}
git rm -r python 2>/dev/null || rmdir python/experiments python 2>/dev/null
```

(If `git rm -r python` fails because the dir isn't tracked anymore, use plain `rm -rf python`.)

- [ ] **Step 10: Verify each notebook still opens and runs**

For each notebook:
```bash
cd experiments/passive_impact/load_experiments
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb
diff <(jq -r '.cells[].outputs' analysis.ipynb 2>/dev/null) <(jq -r '.cells[].outputs' analysis_executed.ipynb 2>/dev/null) | head
rm analysis_executed.ipynb
cd -
```

Expected: notebook executes without errors. If outputs differ visually that's OK; if cells throw exceptions (e.g. `FileNotFoundError`), the data path repoint missed something.

Repeat for `experiments/agressive_impact/load_experiments/analysis.ipynb`.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "Migrate notebooks + plot utilities into experiments/ tree"
```

---

## Task 10: Create new `queue_simulation_efficient` Rust binary

**Files:**
- Create: `code/src/bin/queue_simulation/efficient.rs`
- Modify: `code/src/Cargo.toml` (add `[[bin]]` entry)

This binary mirrors `single_queue_efficient_with_us` but skips `TailImpact`/impact aggregation and writes only the queue paths.

- [ ] **Step 1: Create the binary file**

`code/src/bin/queue_simulation/efficient.rs`:

```rust
//! queue_simulation_efficient — counterfactual queue paths in presence of a metaorder.
//!
//! Same underlying simulation as single_queue_efficient_with_us but without
//! computing the impact curve.

use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{simulate, simulate_with_externals, ConditionalSimulationContext};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events, create_meta_orders,
    extract_events_by_dim, sample_queue_at_times,
};
use simulation_project::utils::{write_npy_f64_1d, write_npy_u32};

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let t_total = Instant::now();

    // ===== Configuration =====
    let time_horizon = 100.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 200;

    let a_l = 100.0;
    let b_l = -0.275;
    let a_c = 2.0;
    let b_c = 0.125;

    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    let n_meta: u32 = 375;
    let meta_start = 1.0;
    let meta_end = 4.0 * time_horizon / 5.0;

    println!("=== Queue Simulation (single, efficient) ===");

    // ===== Pre-simulate Hawkes =====
    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu, alpha.clone(), beta.clone(),
    );
    let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
    let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);

    // ===== Simulate baseline q =====
    let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);
    let q_result_internal = simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(42));
    let q_result = merge_events(&q_result_internal, &hawkes_as_market);
    let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);

    // ===== Build evaluation grid (uniform in time) =====
    let n_times = 1000usize;
    let times: Vec<f64> = (0..n_times)
        .map(|i| i as f64 * time_horizon / (n_times as f64 - 1.0))
        .collect();

    // ===== Conditioning + externals =====
    let q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));
    let meta_orders = create_meta_orders(n_meta, meta_start, meta_end);
    let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);

    // ===== Run parallel conditional simulations =====
    let q_at_times = sample_queue_at_times(&q_path, &times);

    let t0 = Instant::now();
    let bar_q_paths: Vec<Vec<u32>> = (0..n_simulations).into_par_iter().map(|sim_idx| {
        let ctx = ConditionalSimulationContext::new(
            &process,
            &q_events_by_dim,
            Some(&hawkes_as_market),
            Some(&bar_q_external),
            time_horizon,
        );
        ctx.simulate_queue_at_times(&times, initial_queue_size, None, Some(sim_idx as u64))
    }).collect();
    println!("[TIMING] {} parallel simulations: {:?}", n_simulations, t0.elapsed());

    // ===== Output =====
    let output_dir = "experiments/queue_simulation/load_experiments/data/single/efficient";
    std::fs::create_dir_all(output_dir).unwrap();

    // Queue paths: (n_times, n_simulations + 1) — first column = q, rest = bar_q_sim_i
    let queue_data: Vec<u32> = (0..n_times).flat_map(|t_idx| {
        std::iter::once(q_at_times[t_idx])
            .chain(bar_q_paths.iter().map(move |bar_q| bar_q[t_idx]))
    }).collect();
    write_npy_u32(&format!("{}/queue_paths.npy", output_dir), &queue_data, n_times, n_simulations + 1).unwrap();
    write_npy_f64_1d(&format!("{}/times.npy", output_dir), &times).unwrap();

    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
```

- [ ] **Step 2: Add the `[[bin]]` entry to `code/src/Cargo.toml`**

Append at the bottom:
```toml
[[bin]]
name = "queue_simulation_efficient"
path = "bin/queue_simulation/efficient.rs"
```

- [ ] **Step 3: Build and run**

```bash
cargo build --release --bin queue_simulation_efficient 2>&1 | tail -5
cargo run --release --bin queue_simulation_efficient
ls experiments/queue_simulation/load_experiments/data/single/efficient/
```
Expected: `queue_paths.npy` and `times.npy` are written.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Add queue_simulation_efficient binary (counterfactual queue paths only)"
```

---

## Task 11: Create `queue_simulation` notebook + plot_utils

**Files:**
- Create: `experiments/queue_simulation/load_experiments/plot_utils.py`
- Create: `experiments/queue_simulation/load_experiments/analysis.ipynb`

The plot utilities mirror `passive_impact`'s queue-shade plot but drop everything related to impact.

- [ ] **Step 1: Create plot_utils**

`experiments/queue_simulation/load_experiments/plot_utils.py`:

```python
"""Plot utilities for queue-only counterfactual experiments."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(mode='single', data_mode='efficient'):
    base = f'./data/{mode}/{data_mode}'
    times = np.load(f'{base}/times.npy')
    queue = np.load(f'{base}/queue_paths.npy')
    n_sims = queue.shape[1] - 1
    df = pd.DataFrame(
        queue,
        index=pd.Index(times, name='time'),
        columns=['q'] + [f'bar_q_sim_{i}' for i in range(n_sims)],
    )
    return df


def plot_queue_shades(df, meta_end=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sim_cols = [c for c in df.columns if c.startswith('bar_q_sim_')]
    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.05, linewidth=0.5)
    avg = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg, color='red', linewidth=2.5, label='Mean $\\bar{q}$')
    ax.plot(df.index, df['q'], color='black', linewidth=2.5, label='q (baseline)')
    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Queue size')
    ax.set_title('Counterfactual queue paths under metaorder')
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {save_path}')
    else:
        plt.show()


def generate_all_plots(mode='single', data_mode='efficient', meta_end=80.0):
    df = load_data(mode, data_mode)
    plot_queue_shades(df, meta_end=meta_end, save_path=f'images/queue_paths_{mode}.png')


if __name__ == '__main__':
    generate_all_plots()
```

- [ ] **Step 2: Create the notebook (small, just calls plot_utils)**

The simplest way: write a minimal `analysis.ipynb` that has two cells. Use `jupyter nbconvert` from a Python script, or hand-write JSON. Easier: copy `experiments/passive_impact/load_experiments/analysis.ipynb` and strip cells until only the queue-shade cell remains.

```bash
cp experiments/passive_impact/load_experiments/analysis.ipynb \
   experiments/queue_simulation/load_experiments/analysis.ipynb
```

Then open the copy in jupyter, delete every cell except the one that loads queue data and plots queue shades, save.

If editing JSON by hand is preferred, here is the minimal notebook:

```json
{
 "cells": [
  {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": null,
   "source": ["from plot_utils import load_data, plot_queue_shades\n",
              "df = load_data(mode='single', data_mode='efficient')\n",
              "plot_queue_shades(df, meta_end=80.0)"]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
 "nbformat": 4, "nbformat_minor": 5
}
```

- [ ] **Step 3: Verify the notebook runs**

```bash
cd experiments/queue_simulation/load_experiments
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb
ls -la analysis_executed.ipynb
rm analysis_executed.ipynb
cd -
```
Expected: executes cleanly.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Add queue_simulation notebook + plot_utils"
```

---

## Task 12: `passive_impact` facade + `custom_experiment/main.py`

**Files:**
- Create: `code/python/simproj/passive_impact.py`
- Create: `experiments/passive_impact/custom_experiment/main.py`
- Create: `experiments/passive_impact/custom_experiment/README.md`
- Create: `code/python/tests/test_passive_impact_facade.py`
- Modify: `code/python/simproj/__init__.py`

The facade composes the primitives bound in Tasks 5-8 to reproduce the same outputs as the existing `single_queue_efficient_with_us` and `..._without_us` binaries, parameterised by a `PassiveImpactConfig` dataclass.

- [ ] **Step 1: Create the facade module**

`code/python/simproj/passive_impact.py`:

```python
"""Facade for the passive impact experiment.

Wraps the bound primitives (Hawkes, AffineQueueProcess, ConditionalSimulationContext,
TailImpact) into a single `run(config)` entry point that returns the same arrays the
Rust binaries write today.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from . import _native


@dataclass
class PassiveImpactConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500
    initial_queue_size: int = 200
    mode: str = "single"            # "single" | "double"
    side: str = "both"              # "with" | "without" | "both"

    # Hawkes
    mu: float = 1.0
    alpha: list = field(default_factory=lambda: [0.065, 0.2, 0.325, 0.65])
    beta: list = field(default_factory=lambda: [0.15, 0.60, 2.5, 10.0])

    # Affine queue (single-mode params; double-mode uses these for ask side and mirrors for bid)
    a_l: float = 100.0
    b_l: float = -0.275
    a_c: float = 2.0
    b_c: float = 0.125

    # Metaorder: int → evenly-spaced inside metaorder_window;
    #            list[float] / np.ndarray → explicit times (window ignored)
    metaorder: Union[int, list, np.ndarray] = 375
    metaorder_window: tuple = (1.0, 80.0)

    seed: int = 42


def _make_hawkes(cfg: PassiveImpactConfig):
    return _native.MultiExponentialHawkes.with_stationary_state(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
    )


def _make_meta_orders(cfg: PassiveImpactConfig):
    if isinstance(cfg.metaorder, int):
        return _native.create_meta_orders(cfg.metaorder, *cfg.metaorder_window)
    times = np.asarray(cfg.metaorder, dtype=np.float64)
    return _native.create_meta_orders_from_times(times, target_dim=2, total_dims=3)


def _run_single_side(cfg, side: str) -> dict:
    """Run one side ("with" → bar_q given q, "without" → q given bar_q) for single queue."""
    hawkes = _make_hawkes(cfg)
    market = _native.simulate_hawkes_as_market_orders(hawkes, cfg.time_horizon, cfg.seed)
    process = _native.AffineQueueProcess.new_queue(
        float(cfg.initial_queue_size), cfg.a_l, cfg.b_l, cfg.a_c, cfg.b_c,
    )
    q_events = _native.simulate_with_externals(process, cfg.time_horizon, market, cfg.seed)
    full_q = _native.merge_events(q_events, market)

    market_times = market.times()
    cond_by_dim = [list(arr) for arr in _native.extract_events_by_dim(q_events, 3, 2)]

    meta = _make_meta_orders(cfg)
    bar_q_externals = _native.merge_events(meta, market)

    ctx = _native.ConditionalSimulationContext(
        process, cond_by_dim,
        cond_externals=market, new_externals=bar_q_externals,
        t_max=cfg.time_horizon,
    )

    tail = _native.TailImpact.from_affine_queue(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
        cfg.b_l, cfg.b_c, list(market_times),
    )

    q_at_market = _native.sample_queue_at_times(full_q, cfg.initial_queue_size, market_times)

    queue_paths = np.empty((len(market_times), cfg.n_simulations + 1), dtype=np.uint32)
    queue_paths[:, 0] = q_at_market
    impact_paths = np.empty((len(market_times), cfg.n_simulations), dtype=np.float64)
    # NOTE: For brevity this skeleton runs the conditional simulation in Python
    # serially. Production parity with the Rust binaries' rayon-parallel path is
    # achieved by adding a parallel runner in PyO3 (TailImpact-aware) — see
    # the open-implementation note in the spec. For the first cut serial is acceptable.
    for sim_idx in range(cfg.n_simulations):
        bar_q_samples = ctx.simulate_queue_at_times(
            market_times, cfg.initial_queue_size, seed=sim_idx,
        )
        queue_paths[:, sim_idx + 1] = bar_q_samples
        # Impact computation goes here; see TailImpact docs for the formula.
        # Placeholder writes zeros — the full impact loop is wired in Task 13.
        impact_paths[:, sim_idx] = 0.0

    return {
        "times": np.asarray(market_times, dtype=np.float64),
        "queue_paths": queue_paths,
        "impact_paths": impact_paths,
    }


def run(config: PassiveImpactConfig) -> dict:
    """Run the passive impact experiment per `config`.

    Returns a dict with the same arrays the Rust binaries write today.
    For mode="single" + side="both", returns a dict with keys
    {"with": {...}, "without": {...}} each holding the per-side result dict.
    """
    if config.mode == "double":
        raise NotImplementedError("double-queue facade — wired in Task 14")
    if config.side == "both":
        return {
            "with": _run_single_side(config, "with"),
            "without": _run_single_side(config, "without"),
        }
    return _run_single_side(config, config.side)


def save(result: dict, output_dir: str) -> None:
    """Persist a result (or both-side dict) as .npy files under output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    if "with" in result and "without" in result:
        save(result["with"], os.path.join(output_dir, "with"))
        save(result["without"], os.path.join(output_dir, "without"))
        return
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
```

(Note the explicit incomplete-impact placeholder at the bottom of `_run_single_side`. Task 13 wires real impact computation; this task's smoke test does not assert impact-path correctness, only shape.)

- [ ] **Step 2: Re-export the facade from `simproj/__init__.py`**

Add the line:
```python
from . import passive_impact
```

- [ ] **Step 3: Write the smoke test**

`code/python/tests/test_passive_impact_facade.py`:

```python
import numpy as np
from simproj.passive_impact import PassiveImpactConfig, run, save


def test_passive_impact_smoke(tmp_path):
    cfg = PassiveImpactConfig(
        time_horizon=2.0,
        n_simulations=2,
        initial_queue_size=200,
        mode="single",
        side="with",
        metaorder=4,
        metaorder_window=(0.1, 1.5),
        seed=42,
    )
    result = run(cfg)
    assert "times" in result
    assert "queue_paths" in result
    assert "impact_paths" in result
    assert result["queue_paths"].shape[1] == 3  # q + 2 sims
    assert result["impact_paths"].shape[1] == 2

    save(result, str(tmp_path))
    assert (tmp_path / "times.npy").exists()
    assert (tmp_path / "queue_paths.npy").exists()
    assert (tmp_path / "impact_paths.npy").exists()
```

```bash
cd code/python && pytest tests/test_passive_impact_facade.py -v && cd -
```
Expected: 1 passed.

- [ ] **Step 4: Create `experiments/passive_impact/custom_experiment/main.py`**

```python
"""Run a parameter-tweaked passive impact experiment."""
from simproj import passive_impact as pi

# ──────────────── CONFIG ────────────────
config = pi.PassiveImpactConfig(
    time_horizon=100.0,
    n_simulations=500,
    initial_queue_size=200,
    mode="single",                # "single" | "double"
    side="both",                  # "with" | "without" | "both"
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    # Metaorder accepts:
    #   int N         → N evenly-spaced orders inside metaorder_window
    #   list/ndarray  → explicit list of arrival times (window ignored)
    metaorder=375,
    metaorder_window=(1.0, 80.0),
    # Example explicit list (commented out):
    # metaorder=[1.0, 2.5, 4.0, 7.0, 12.0, 30.0, 60.0],
    seed=42,
)
# ────────────────────────────────────────

if __name__ == "__main__":
    result = pi.run(config)
    pi.save(result, "experiments/passive_impact/custom_experiment/output/")
    print("Done. Outputs in experiments/passive_impact/custom_experiment/output/")
```

- [ ] **Step 5: Create the README**

`experiments/passive_impact/custom_experiment/README.md`:

```markdown
# Custom passive impact experiment

Edit the `config = pi.PassiveImpactConfig(...)` block in `main.py`, then run:

    python experiments/passive_impact/custom_experiment/main.py

Outputs land in `output/` (gitignored).
```

- [ ] **Step 6: Smoke-run main.py with a tiny config**

Temporarily edit `main.py` to set `n_simulations=2, time_horizon=2.0, metaorder=4, metaorder_window=(0.1, 1.5)`, then:
```bash
python experiments/passive_impact/custom_experiment/main.py
ls experiments/passive_impact/custom_experiment/output/
```
Expected: prints "Done." and `with/`, `without/` subdirs each contain 3 `.npy` files.

Revert the config to the production defaults (don't commit the smoke values).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Add passive_impact facade + custom_experiment main.py (impact stub)"
```

---

## Task 13: Wire real impact computation into `passive_impact` facade

**Files:**
- Modify: `code/python/src/lib.rs` — bind a free function `compute_impact_path` that wraps `ImpactPath::new`
- Modify: `code/python/simproj/__init__.py`
- Modify: `code/python/simproj/passive_impact.py` — replace placeholder
- Modify: `code/python/tests/test_passive_impact_facade.py` — strengthen impact assertion

The right primitive is `ImpactPath::new(q: QueuePath, q_bar: QueuePath, tail_impact: &TailImpact)` in [`code/src/conditional_impact/flow_imbalance_model/single_queue/impact_path.rs`](../../code/src/conditional_impact/flow_imbalance_model/single_queue/impact_path.rs). It returns an `ImpactPath { impact_path: Vec<f64> }`. The free PyO3 wrapper takes the two queue event streams (already exposed as `PySimulationResult`) plus `initial_q`, converts each to `QueuePath` via `AffineQueueProcess::result_to_queue_path`, then calls `ImpactPath::new`.

- [ ] **Step 1: Add the binding to `code/python/src/lib.rs`**

Add the import:
```rust
use simulation_project::conditional_impact::ImpactPath;
```

Append the free function:

```rust
/// Compute the impact path I(t) for a (q, bar_q) pair via the affine-queue model.
///
/// q_events / bar_q_events are the full event streams of the two queue processes
/// (e.g. as returned by merge_events of conditional simulations + market orders).
/// initial_q is the starting queue size used for both paths.
#[pyfunction]
fn compute_impact_path<'py>(
    py: Python<'py>,
    q_events: &PySimulationResult,
    bar_q_events: &PySimulationResult,
    initial_q: u32,
    tail_impact: &PyTailImpact,
) -> &'py PyArray1<f64> {
    let q_path = AffineQueueProcess::result_to_queue_path(&q_events.inner, initial_q);
    let bar_q_path = AffineQueueProcess::result_to_queue_path(&bar_q_events.inner, initial_q);
    let impact = ImpactPath::new(q_path, bar_q_path, &tail_impact.inner);
    impact.impact_path.into_pyarray(py)
}
```

Register in `#[pymodule]`:
```rust
m.add_function(wrap_pyfunction!(compute_impact_path, m)?)?;
```

- [ ] **Step 2: Re-export**

In `simproj/__init__.py`:
```python
compute_impact_path = _native.compute_impact_path
```

- [ ] **Step 3: Rebuild**

```bash
cd code/python && maturin develop --release && cd -
```

- [ ] **Step 4: Wire into the facade**

In `code/python/simproj/passive_impact.py`, the `_run_single_side` function currently zero-fills `impact_paths[:, sim_idx]`. Replace the loop body so each iteration also produces the `bar_q` event stream needed by `compute_impact_path`. The structural change:

(a) Before the loop, capture `bar_q_external` (already exists) and the conditioning context.

(b) Inside the loop, instead of calling `simulate_queue_at_times` (which only returns sampled values), call `ctx.simulate(seed=sim_idx)` to get the full `PySimulationResult` of bar_q events. Then merge with market orders, sample queue at market times for `queue_paths`, and pass the merged event stream to `compute_impact_path` for the impact column.

Replace the loop in `_run_single_side` with:

```python
    for sim_idx in range(cfg.n_simulations):
        bar_q_events = ctx.simulate(seed=sim_idx)
        bar_q_full = _native.merge_events(bar_q_events, market)
        bar_q_samples_at_market = _native.sample_queue_at_times(
            bar_q_full, cfg.initial_queue_size, market_times,
        )
        queue_paths[:, sim_idx + 1] = bar_q_samples_at_market
        impact_paths[:, sim_idx] = _native.compute_impact_path(
            full_q, bar_q_full, cfg.initial_queue_size, tail,
        )
```

(`full_q` is already computed earlier in `_run_single_side` — it is the merged baseline q events including market orders.)

- [ ] **Step 5: Strengthen the smoke test**

Edit `code/python/tests/test_passive_impact_facade.py` and add to the existing `test_passive_impact_smoke`:

```python
    # Impact paths should be non-trivial (non-zero somewhere)
    assert np.any(result["impact_paths"] != 0.0)
```

```bash
cd code/python && pytest tests/test_passive_impact_facade.py -v && cd -
```
Expected: 1 passed.

- [ ] **Step 5: Strengthen the smoke test**

Edit `code/python/tests/test_passive_impact_facade.py` and add to the existing test:
```python
    # Impact paths should be non-trivial (non-zero somewhere)
    assert np.any(result["impact_paths"] != 0.0)
```

```bash
cd code/python && maturin develop --release && pytest tests/test_passive_impact_facade.py -v && cd -
```
Expected: 1 passed.

- [ ] **Step 6: End-to-end parity check vs. Rust binary**

Run the facade with the same defaults as the existing `single_queue_efficient_with_us` binary and compare the output to the migrated baseline data:

```python
# code/python/tests/test_passive_parity.py
import numpy as np
from simproj.passive_impact import PassiveImpactConfig, run

def test_facade_matches_rust_binary_baseline():
    cfg = PassiveImpactConfig()  # defaults match the rust binary
    cfg.side = "with"
    result = run(cfg)
    baseline_q = np.load(
        "../../experiments/passive_impact/load_experiments/data/single/efficient/with/queue_paths.npy"
    )
    assert result["queue_paths"].shape == baseline_q.shape
    # Means should be close (path ordering may differ due to parallel/serial dispatch)
    assert np.isclose(result["queue_paths"].mean(), baseline_q.mean(), rtol=0.02)
```

```bash
cd code/python && pytest tests/test_passive_parity.py -v && cd -
```
Expected: passes (with rtol=2% for distributional comparison; exact byte-equality is unrealistic when serial-Python ordering differs from rayon-parallel).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Wire real impact computation into passive_impact facade"
```

---

## Task 14: `agressive_impact` facade + `custom_experiment/main.py`

**Files:**
- Create: `code/python/simproj/agressive_impact.py`
- Create: `experiments/agressive_impact/custom_experiment/main.py`
- Create: `experiments/agressive_impact/custom_experiment/README.md`
- Create: `code/python/tests/test_agressive_impact_facade.py`
- Modify: `code/python/simproj/__init__.py`

Mirrors the existing `agressive_impact` and `agressive_impact_hybrid` Rust binaries, with a `model = "propagator" | "hybrid"` config knob.

- [ ] **Step 1: Create `code/python/simproj/agressive_impact.py`**

```python
"""Facade for the aggressive impact experiment (propagator + hybrid models)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from . import _native


@dataclass
class AggressiveImpactConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500
    initial_queue_size: int = 200

    model: str = "propagator"       # "propagator" | "hybrid"
    bar_kappa: Optional[float] = None  # required when model == "hybrid"

    # Hawkes
    mu: float = 1.0
    alpha: list = field(default_factory=lambda: [0.065, 0.2, 0.325, 0.65])
    beta: list = field(default_factory=lambda: [0.15, 0.60, 2.5, 10.0])

    # Affine queue
    a_l: float = 100.0
    b_l: float = -0.275
    a_c: float = 2.0
    b_c: float = 0.125

    # Kappa(q) — defaults to the paper's c1 * sqrt(log(e^{-c2*q} + 1))
    kappa: Callable[[float], float] = field(
        default_factory=lambda: lambda q: 1000.0 * (np.log(np.exp(-0.01 * q) + 1.0) ** 0.5)
    )

    # Metaorder (aggressive metaorders are dim=2 = market orders)
    metaorder: Union[int, list, np.ndarray] = 200
    metaorder_window: tuple = (1.0, 75.0)

    seed: int = 42


def _make_meta_orders(cfg: AggressiveImpactConfig):
    if isinstance(cfg.metaorder, int):
        meta = _native.create_meta_orders(cfg.metaorder, *cfg.metaorder_window)
        return _native.events_to_dim(meta, target_dim=2, total_dims=3)
    times = np.asarray(cfg.metaorder, dtype=np.float64)
    return _native.create_meta_orders_from_times(times, target_dim=2, total_dims=3)


def run(cfg: AggressiveImpactConfig) -> dict:
    if cfg.model == "hybrid" and cfg.bar_kappa is None:
        raise ValueError("hybrid model requires bar_kappa")

    hawkes = _native.MultiExponentialHawkes.with_stationary_state(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
    )
    market = _native.simulate_hawkes_as_market_orders(hawkes, cfg.time_horizon, cfg.seed)
    process = _native.AffineQueueProcess.new_queue(
        float(cfg.initial_queue_size), cfg.a_l, cfg.b_l, cfg.a_c, cfg.b_c,
    )
    q_events = _native.simulate_with_externals(process, cfg.time_horizon, market, cfg.seed)
    q_full = _native.merge_events(q_events, market)
    cond_by_dim = [list(arr) for arr in _native.extract_events_by_dim(q_events, 3, 2)]

    meta = _make_meta_orders(cfg)
    market_times = list(market.times())
    meta_times = list(meta.times())

    eval_entries = sorted([(t, True) for t in market_times] + [(t, False) for t in meta_times])
    eval_times = np.array([t for t, _ in eval_entries], dtype=np.float64)
    is_market_order = [b for _, b in eval_entries]
    bar_q_external = _native.merge_events(meta, market)

    q_at_eval = _native.sample_queue_at_times(q_full, cfg.initial_queue_size, eval_times)

    n_times = len(eval_times)
    queue_paths = np.empty((n_times, cfg.n_simulations + 1), dtype=np.uint32)
    queue_paths[:, 0] = q_at_eval
    impact_paths = np.empty((n_times, cfg.n_simulations), dtype=np.float64)

    for sim_idx in range(cfg.n_simulations):
        ctx = _native.ConditionalSimulationContext(
            process, cond_by_dim,
            cond_externals=market, new_externals=bar_q_external,
            t_max=cfg.time_horizon,
        )
        bar_q = ctx.simulate_queue_at_times(eval_times, cfg.initial_queue_size, seed=sim_idx)
        queue_paths[:, sim_idx + 1] = bar_q

        # NOTE: hybrid model uses a different Rust function path. Treat propagator
        # as the default; hybrid wiring requires binding from_queue_samples_hybrid
        # in PyO3 (small follow-up). For now hybrid raises NotImplementedError.
        if cfg.model == "hybrid":
            raise NotImplementedError(
                "hybrid model wiring requires binding from_queue_samples_hybrid; "
                "see Task 15."
            )

        result = _native.aggressive_impact_from_queue_samples(
            q_samples=q_at_eval, q_bar_samples=bar_q,
            eval_times=eval_times, is_market_order=is_market_order,
            hawkes=hawkes, kappa=cfg.kappa,
        )
        impact_paths[:, sim_idx] = result.impact()

    return {
        "times": eval_times,
        "queue_paths": queue_paths,
        "impact_paths": impact_paths,
        "event_types": np.array([1.0 if b else 0.0 for b in is_market_order]),
    }


def save(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
```

- [ ] **Step 2: Re-export from `__init__.py`**

```python
from . import agressive_impact
```

- [ ] **Step 3: Smoke test**

`code/python/tests/test_agressive_impact_facade.py`:

```python
import numpy as np
from simproj.agressive_impact import AggressiveImpactConfig, run, save


def test_agressive_impact_propagator_smoke(tmp_path):
    cfg = AggressiveImpactConfig(
        time_horizon=2.0, n_simulations=2, initial_queue_size=200,
        model="propagator",
        metaorder=4, metaorder_window=(0.1, 1.5),
        seed=42,
    )
    result = run(cfg)
    assert result["queue_paths"].shape[1] == 3
    assert result["impact_paths"].shape[1] == 2
    assert "event_types" in result

    save(result, str(tmp_path))
    assert (tmp_path / "queue_paths.npy").exists()
    assert (tmp_path / "event_types.npy").exists()
```

```bash
cd code/python && pytest tests/test_agressive_impact_facade.py -v && cd -
```
Expected: 1 passed.

- [ ] **Step 4: Create main.py**

`experiments/agressive_impact/custom_experiment/main.py`:

```python
"""Run a parameter-tweaked aggressive impact experiment."""
import numpy as np

from simproj import agressive_impact as ai

# ──────────────── CONFIG ────────────────
config = ai.AggressiveImpactConfig(
    time_horizon=100.0,
    n_simulations=500,
    initial_queue_size=200,
    model="propagator",          # "propagator" | "hybrid"
    bar_kappa=None,              # required when model == "hybrid"
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    # kappa defaults to c1 * sqrt(log(e^{-c2 q} + 1)); override if needed:
    # kappa=lambda q: 800.0 * np.sqrt(q),
    metaorder=200,
    metaorder_window=(1.0, 75.0),
    seed=42,
)
# ────────────────────────────────────────

if __name__ == "__main__":
    result = ai.run(config)
    ai.save(result, "experiments/agressive_impact/custom_experiment/output/")
    print("Done. Outputs in experiments/agressive_impact/custom_experiment/output/")
```

- [ ] **Step 5: Create README**

`experiments/agressive_impact/custom_experiment/README.md`:

```markdown
# Custom aggressive impact experiment

Edit `config` in `main.py`, then:

    python experiments/agressive_impact/custom_experiment/main.py

For the hybrid model, set `model="hybrid"` and provide a `bar_kappa` (a constant
weight on the propagator term). Outputs land in `output/` (gitignored).
```

- [ ] **Step 6: Smoke-run with tiny config**

Temporarily edit `main.py` to `n_simulations=2, time_horizon=2.0, metaorder=4, metaorder_window=(0.1, 1.5)`:
```bash
python experiments/agressive_impact/custom_experiment/main.py
ls experiments/agressive_impact/custom_experiment/output/
```
Expected: 4 .npy files. Revert config.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Add agressive_impact facade + custom_experiment main.py (propagator only; hybrid stub)"
```

---

## Task 15: Wire `hybrid` aggressive impact model

**Files:**
- Modify: `code/python/src/lib.rs`
- Modify: `code/python/simproj/__init__.py`
- Modify: `code/python/simproj/agressive_impact.py`
- Modify: `code/python/tests/test_agressive_impact_facade.py`

`AggressiveImpactPath::from_queue_samples_hybrid` already exists in Rust (see [`code/src/conditional_impact/propagator_model/mod.rs:141`](../../code/src/conditional_impact/propagator_model/mod.rs)). Bind it in PyO3 and dispatch from the facade.

- [ ] **Step 1: Add the hybrid binding to `code/python/src/lib.rs`**

```rust
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn aggressive_impact_from_queue_samples_hybrid(
    py: Python,
    q_samples: PyReadonlyArray1<u32>,
    q_bar_samples: PyReadonlyArray1<u32>,
    eval_times: PyReadonlyArray1<f64>,
    is_market_order: Vec<bool>,
    hawkes: &PyMultiExponentialHawkes,
    kappa: PyObject,
    bar_kappa: f64,
    b_l: f64,
    b_c: f64,
) -> PyResult<PyAggressiveImpactPath> {
    let kappa_clone = kappa.clone_ref(py);
    let path = AggressiveImpactPath::from_queue_samples_hybrid(
        q_samples.as_slice().unwrap(),
        q_bar_samples.as_slice().unwrap(),
        eval_times.as_slice().unwrap(),
        &is_market_order,
        &hawkes.inner,
        |q: f64| -> f64 {
            Python::with_gil(|py| {
                let res = kappa_clone.call1(py, (q,)).unwrap();
                res.extract::<f64>(py).unwrap()
            })
        },
        bar_kappa,
        b_l,
        b_c,
    );
    Ok(PyAggressiveImpactPath { impact_path: path.impact_path })
}
```

(Verify the Rust signature — read `code/src/conditional_impact/propagator_model/mod.rs:141` and adjust the parameter list to match exactly. The above mirrors the propagator binding plus `bar_kappa, b_l, b_c`. If the Rust function takes different extra args, update the Python signature accordingly.)

Register in `#[pymodule]`:
```rust
m.add_function(wrap_pyfunction!(aggressive_impact_from_queue_samples_hybrid, m)?)?;
```

- [ ] **Step 2: Re-export**

In `simproj/__init__.py` add:
```python
aggressive_impact_from_queue_samples_hybrid = _native.aggressive_impact_from_queue_samples_hybrid
```

- [ ] **Step 3: Replace the NotImplementedError in the facade**

In `code/python/simproj/agressive_impact.py`, replace:
```python
        if cfg.model == "hybrid":
            raise NotImplementedError(...)

        result = _native.aggressive_impact_from_queue_samples(...)
```

with:
```python
        if cfg.model == "hybrid":
            result = _native.aggressive_impact_from_queue_samples_hybrid(
                q_samples=q_at_eval, q_bar_samples=bar_q,
                eval_times=eval_times, is_market_order=is_market_order,
                hawkes=hawkes, kappa=cfg.kappa,
                bar_kappa=cfg.bar_kappa,
                b_l=cfg.b_l, b_c=cfg.b_c,
            )
        else:
            result = _native.aggressive_impact_from_queue_samples(
                q_samples=q_at_eval, q_bar_samples=bar_q,
                eval_times=eval_times, is_market_order=is_market_order,
                hawkes=hawkes, kappa=cfg.kappa,
            )
```

- [ ] **Step 4: Add hybrid smoke test**

Append to `code/python/tests/test_agressive_impact_facade.py`:

```python
def test_agressive_impact_hybrid_smoke(tmp_path):
    cfg = AggressiveImpactConfig(
        time_horizon=2.0, n_simulations=2, initial_queue_size=200,
        model="hybrid", bar_kappa=10.0,
        metaorder=4, metaorder_window=(0.1, 1.5),
        seed=42,
    )
    result = run(cfg)
    assert result["impact_paths"].shape[1] == 2
```

```bash
cd code/python && maturin develop --release && pytest tests/test_agressive_impact_facade.py -v && cd -
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Bind hybrid aggressive impact model and wire into facade"
```

---

## Task 16: `queue_simulation` facade + `custom_experiment/main.py`

**Files:**
- Create: `code/python/simproj/queue_simulation.py`
- Create: `experiments/queue_simulation/custom_experiment/main.py`
- Create: `experiments/queue_simulation/custom_experiment/README.md`
- Create: `code/python/tests/test_queue_simulation_facade.py`
- Modify: `code/python/simproj/__init__.py`

Simplest of the three facades — no impact computation, just queue paths.

- [ ] **Step 1: Create `code/python/simproj/queue_simulation.py`**

```python
"""Facade for queue-only counterfactual simulation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from . import _native


@dataclass
class QueueSimulationConfig:
    time_horizon: float = 100.0
    n_simulations: int = 500
    n_eval_times: int = 1000
    initial_queue_size: int = 200
    mode: str = "single"            # "single" | "double"

    mu: float = 1.0
    alpha: list = field(default_factory=lambda: [0.065, 0.2, 0.325, 0.65])
    beta: list = field(default_factory=lambda: [0.15, 0.60, 2.5, 10.0])

    a_l: float = 100.0
    b_l: float = -0.275
    a_c: float = 2.0
    b_c: float = 0.125

    metaorder: Union[int, list, np.ndarray] = 375
    metaorder_window: tuple = (1.0, 80.0)

    seed: int = 42


def run(cfg: QueueSimulationConfig) -> dict:
    if cfg.mode == "double":
        raise NotImplementedError("double-queue queue_simulation — follow-up")

    hawkes = _native.MultiExponentialHawkes.with_stationary_state(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
    )
    market = _native.simulate_hawkes_as_market_orders(hawkes, cfg.time_horizon, cfg.seed)
    process = _native.AffineQueueProcess.new_queue(
        float(cfg.initial_queue_size), cfg.a_l, cfg.b_l, cfg.a_c, cfg.b_c,
    )
    q_events = _native.simulate_with_externals(process, cfg.time_horizon, market, cfg.seed)
    q_full = _native.merge_events(q_events, market)
    cond_by_dim = [list(arr) for arr in _native.extract_events_by_dim(q_events, 3, 2)]

    if isinstance(cfg.metaorder, int):
        meta = _native.create_meta_orders(cfg.metaorder, *cfg.metaorder_window)
    else:
        meta = _native.create_meta_orders_from_times(
            np.asarray(cfg.metaorder, dtype=np.float64), target_dim=2, total_dims=3,
        )
    bar_q_external = _native.merge_events(meta, market)

    times = np.linspace(0.0, cfg.time_horizon, cfg.n_eval_times).astype(np.float64)
    q_at_times = _native.sample_queue_at_times(q_full, cfg.initial_queue_size, times)

    queue_paths = np.empty((cfg.n_eval_times, cfg.n_simulations + 1), dtype=np.uint32)
    queue_paths[:, 0] = q_at_times
    for sim_idx in range(cfg.n_simulations):
        ctx = _native.ConditionalSimulationContext(
            process, cond_by_dim,
            cond_externals=market, new_externals=bar_q_external,
            t_max=cfg.time_horizon,
        )
        queue_paths[:, sim_idx + 1] = ctx.simulate_queue_at_times(
            times, cfg.initial_queue_size, seed=sim_idx,
        )

    return {"times": times, "queue_paths": queue_paths}


def save(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, arr in result.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
```

- [ ] **Step 2: Re-export**

```python
from . import queue_simulation
```

- [ ] **Step 3: Smoke test**

`code/python/tests/test_queue_simulation_facade.py`:

```python
import numpy as np
from simproj.queue_simulation import QueueSimulationConfig, run, save


def test_queue_simulation_smoke(tmp_path):
    cfg = QueueSimulationConfig(
        time_horizon=2.0, n_simulations=2, n_eval_times=20,
        initial_queue_size=200, mode="single",
        metaorder=4, metaorder_window=(0.1, 1.5), seed=42,
    )
    result = run(cfg)
    assert result["queue_paths"].shape == (20, 3)
    save(result, str(tmp_path))
    assert (tmp_path / "queue_paths.npy").exists()
```

```bash
cd code/python && pytest tests/test_queue_simulation_facade.py -v && cd -
```
Expected: 1 passed.

- [ ] **Step 4: Create main.py**

`experiments/queue_simulation/custom_experiment/main.py`:

```python
"""Run a parameter-tweaked queue-only counterfactual simulation."""
from simproj import queue_simulation as qs

# ──────────────── CONFIG ────────────────
config = qs.QueueSimulationConfig(
    time_horizon=100.0,
    n_simulations=500,
    n_eval_times=1000,
    initial_queue_size=200,
    mode="single",
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    metaorder=375,
    metaorder_window=(1.0, 80.0),
    seed=42,
)
# ────────────────────────────────────────

if __name__ == "__main__":
    result = qs.run(config)
    qs.save(result, "experiments/queue_simulation/custom_experiment/output/")
    print("Done. Outputs in experiments/queue_simulation/custom_experiment/output/")
```

- [ ] **Step 5: Create README**

`experiments/queue_simulation/custom_experiment/README.md`:

```markdown
# Custom queue simulation experiment

Edit `config` in `main.py`, then:

    python experiments/queue_simulation/custom_experiment/main.py

Outputs (`times.npy`, `queue_paths.npy`) land in `output/` (gitignored).
```

- [ ] **Step 6: Smoke-run main.py**

Temporarily edit to a small config, run, verify, revert.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Add queue_simulation facade + custom_experiment main.py"
```

---

## Task 17: Update `README.md` and final cleanup

**Files:**
- Modify: `README.md`
- Modify: `.gitignore` (verify Task 4 entries are present)
- Verify: legacy `data/` and `python/` directories are gone

- [ ] **Step 1: Update the Experiments section in README.md**

Open `README.md`. Replace the existing Experiments section (the table that lists single_queue_efficient et al.) with:

```markdown
## Experiments

Three top-level experiment categories live under `experiments/`:

| Category | What it shows | Pre-saved baseline | Custom |
|---|---|---|---|
| **Passive Impact** | Conditional impact from limit-order metaorders (single + double queue) | [`experiments/passive_impact/load_experiments/analysis.ipynb`](experiments/passive_impact/load_experiments/analysis.ipynb) | [`experiments/passive_impact/custom_experiment/main.py`](experiments/passive_impact/custom_experiment/main.py) |
| **Aggressive Impact** | Market-order impact under propagator and hybrid models | [`experiments/agressive_impact/load_experiments/analysis.ipynb`](experiments/agressive_impact/load_experiments/analysis.ipynb) | [`experiments/agressive_impact/custom_experiment/main.py`](experiments/agressive_impact/custom_experiment/main.py) |
| **Queue Simulation** | Counterfactual queue paths under a metaorder (no impact curve) | [`experiments/queue_simulation/load_experiments/analysis.ipynb`](experiments/queue_simulation/load_experiments/analysis.ipynb) | [`experiments/queue_simulation/custom_experiment/main.py`](experiments/queue_simulation/custom_experiment/main.py) |

Each `load_experiments/` folder contains the notebook, plot utilities, and committed `.npy` baseline data produced by the Rust binaries. Each `custom_experiment/` folder contains a single `main.py` whose top section is a config dataclass; edit and run.

### Setup

    cd code/python && maturin develop --release && cd -

### Regenerate baselines (Rust binaries)

    cargo run --release --bin single_queue_efficient_with_us
    cargo run --release --bin single_queue_efficient_without_us
    cargo run --release --bin double_queue_efficient_with_us
    cargo run --release --bin double_queue_efficient_without_us
    cargo run --release --bin agressive_impact
    cargo run --release --bin agressive_impact_hybrid
    cargo run --release --bin queue_simulation_efficient

(General variants are also kept for validation: `*_general_with_us`, `*_general_without_us`.)

### Run a custom experiment

    python experiments/passive_impact/custom_experiment/main.py
    # outputs land in experiments/passive_impact/custom_experiment/output/
```

- [ ] **Step 2: Update embedded image paths in README.md**

The existing README has image links like `python/experiments/single_queue_impact/images/...`. Update to `experiments/passive_impact/load_experiments/images/...`.

```bash
sed -i '' 's|python/experiments/single_queue_impact/|experiments/passive_impact/load_experiments/|g' README.md
sed -i '' 's|python/experiments/double_queue_impact/|experiments/passive_impact/load_experiments/|g' README.md
sed -i '' 's|python/experiments/agressive_impact/|experiments/agressive_impact/load_experiments/|g' README.md
```

Spot-check: `grep -n 'python/experiments' README.md` — expected: no matches.

- [ ] **Step 3: Final tree audit**

```bash
ls -d data 2>&1 || echo "ok: data removed"
ls -d python 2>&1 || echo "ok: python removed"
find experiments -name "*.npy" | wc -l
find code -name "*.rs" | wc -l
```
Expected: `data` and `python` are gone; .npy files appear under experiments/.

- [ ] **Step 4: Run all tests**

```bash
cargo test --release 2>&1 | tail -20
cd code/python && pytest tests/ -v && cd -
```
Expected: cargo tests pass; all pytest smoke tests pass.

- [ ] **Step 5: End-to-end manual sanity check**

```bash
# Regenerate one baseline and confirm notebook still loads it cleanly
cargo run --release --bin queue_simulation_efficient
cd experiments/queue_simulation/load_experiments
jupyter nbconvert --to notebook --execute analysis.ipynb --output _check.ipynb
rm _check.ipynb
cd -
```
Expected: cargo binary completes; notebook executes without errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Update README with new experiments structure and final tree cleanup"
```

---

## Validation criteria (from spec)

After all tasks land, manually verify:

1. `cargo run --release --bin <any binary>` from repo root completes and writes into `experiments/<cat>/load_experiments/data/<variant>/`. ✅ (Task 3, Task 10)
2. `cd code/python && maturin develop --release` succeeds. ✅ (Task 4)
3. `python -c "from simproj import passive_impact, agressive_impact, queue_simulation; print('OK')"` succeeds. ✅ (Tasks 12, 14, 16)
4. `python experiments/passive_impact/custom_experiment/main.py` runs and produces .npy + plots in `output/`. ✅ (Task 12; plot generation via `plot_utils` is automatic when integrated — leaving manual plot wiring as a small follow-up if not done in Task 12)
5. Each `experiments/<cat>/load_experiments/analysis.ipynb` runs all cells and reproduces the baseline plots. ✅ (Tasks 9, 11)
6. `cargo test` passes. ✅ (Task 17 verification)
7. `pytest code/python/tests/` passes. ✅ (Task 17 verification)

## Open items deferred to future branches (per spec non-goals)

- Non-linear `reaction_intensity` mode.
- Deeper β-style Rust dedup of single/double queue paths.
- Parallel execution (`rayon`) of the Python facade — currently serial; matches Rust binary at the cost of wall-time. A small follow-up can add a `ParallelFacadeRunner` PyO3 binding.
