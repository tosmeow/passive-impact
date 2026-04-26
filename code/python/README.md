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
