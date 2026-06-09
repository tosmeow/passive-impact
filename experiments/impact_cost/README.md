# Impact Cost

Empirical passive execution-cost experiments for limit orders posted into a
real queue. The workflow anchors on observed queue snapshots, simulates no-us
counterfactual queues, converts queue displacement into passive price impact,
and samples that impact at execution times.

The accounting target is a running execution cost:

```text
C_t = sum_{fills tau_j <= t} qty_j * Delta P_{tau_j-}
```

where `Delta P` is the selected passive impact path and fills use the left-limit
impact convention.

## Main workflow -- `pipelines/`

Run pipeline entry points from the repository root:

```bash
python -m experiments.impact_cost.pipelines.execution_latency_grid --help
python -m experiments.impact_cost.pipelines.impact_cost_pipeline --help
python -m experiments.impact_cost.pipelines.lifecycle_passive_cost_pipeline --help
```

Typical empirical workflow:

1. `execution_latency_grid.py`: infer first-level passive posting/fill labels
   from raw depth snapshots.
2. `impact_cost_pipeline.py`: filter those labels, simulate anchored no-us
   queues, compute passive impact paths, and write fill-level cost
   contributions.
3. `running_cost_diagnostics.py`: optional per-order visual checks for queue
   priority and running cost.

Useful alternatives:

- `queue_pipeline.py`: inspect anchored no-us queue paths directly.
- `scheduled_passive_cost_pipeline.py`: validate a scheduled passive posting
  and synthetic-fill design.
- `lifecycle_passive_cost_pipeline.py`: run looped post/fill/cancel lifecycle
  experiments without requiring exact q1 fill labels.
- `impact_series_pipeline.py`: sample unconditional passive limit sequences and
  summarize impact paths.
- `execution_overlay.py`: draw one-window posting/fill overlays on raw q1.

Common knobs: `raw_side`, `queue_col`, `market_side`, `horizon_seconds`,
`n_simulations`, affine queue parameters `a_l`/`b_l`/`a_c`/`b_c`, and
`impact_model` (`"reduced_form"`, `"tail_propagator"`, or `"structural"`).

Generated CSV/PNG/JSON outputs land in `runs/` and are gitignored except for
[`runs/README.md`](runs/README.md).

## Local data -- `data/`

Default local inputs are:

- `data/raw/2025_05_29_ESM5.parquet`: raw first-level depth/order-flow rows.
- `data/processed/factual_2025_05_29_esm5.parquet`: processed aggregate queue
  input for anchored queue and impact-cost pipelines.

These data files are intentionally gitignored. See [`data/README.md`](data/README.md)
for the expected local layout.

## Components and conventions

Reusable dataframe and accounting helpers live in `core/`; runnable CLI
workflows live in `pipelines/`. Native Rust helpers are exposed through
`simproj` for anchored queue simulation, passive limit selection, fill tracking,
and tail-propagator impact primitives.

For file-by-file details, data schemas, side/sign conventions, native quantity
handling, and pipeline outputs, see [`COMPONENTS.md`](COMPONENTS.md). For the
propagator-input passive tail derivation, see
[`PROPAGATOR_INPUT_PASSIVE_NOTE.md`](PROPAGATOR_INPUT_PASSIVE_NOTE.md).
