# simproj — Python bindings for simulation_project

## Build

From the repository root:

    maturin develop --release --manifest-path code/python/Cargo.toml

Re-run after any change to `code/python/src/lib.rs` or to Rust functions exposed
through PyO3.

## Layout

- `src/lib.rs` — PyO3 wrappers for Rust primitives (compiled into `simproj._native`)
- `simproj/__init__.py` — re-exports
- `simproj/passive_impact.py` — facade for the passive impact experiment
- `simproj/agressive_impact.py` — facade for the aggressive impact experiment
- `simproj/queue_simulation.py` — facade for the queue-only experiment
- `tests/` — pytest smoke tests

## Impact-Cost Native Helpers

The empirical impact-cost workflow lives at `experiments/impact_cost/`, but it
uses native helpers re-exported from `simproj`:

- `simulate_anchored_affine_queue` — simulate no-us queue offsets around an
  empirical queue snapshot path.
- `select_limit_flags_first_every`, `select_limit_flags_indices`, and
  `select_limit_flags_random_fraction` — select passive limit rows for
  counterfactual removal.
- `track_passive_fills` and `simulate_execution_latency_grid` — track passive
  execution under first-level queue priority conventions.
- `passive_flow_impact_from_queue_samples` and
  `passive_tail_propagator_impact_from_queue_samples` — native passive impact
  primitives used by the Python experiment layer.

Run the canonical lifecycle experiment with:

```bash
python -m experiments.impact_cost.load_experiments.lifecycle_passive_cost \
  --config experiments/impact_cost/load_experiments/config.toml
```

See `experiments/impact_cost/README.md` for the side, quantity, and cost
conventions.
