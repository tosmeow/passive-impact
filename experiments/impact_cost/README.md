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

## Main workflow -- `load_experiments/`

The canonical experiment is the lifecycle passive-cost run. Edit
[`load_experiments/config.toml`](load_experiments/config.toml), then run:

```bash
python -m experiments.impact_cost.load_experiments.lifecycle_passive_cost \
  --config experiments/impact_cost/load_experiments/config.toml
```

This final experiment uses empirical consuming-side market-event times and the
factual aggregate queue, then generates our own passive lifecycle. By default,
generated post/cancel intentions are resolved onto actual observed limit/cancel
row quantities on the selected side, while fill/execution times remain random
lifecycle times independent of observed market executions. It is fixed to the
`tail_propagator` impact model; vary lifecycle, episode, queue, and
tail-propagator parameters in `config.toml`.

Generated CSV/JSON outputs land in `load_experiments/data/`; the lifecycle PDF
lands in `load_experiments/images/`. Regenerate the figure from saved summary
outputs with:

```bash
python -m experiments.impact_cost.load_experiments.plot_utils
```

Older raw-fill inference, overlays, scheduled-cost checks, and other diagnostic
scripts live in [`archive/diagnostics/`](archive/diagnostics/). Bulky local
reference materials live under [`archive/references/`](archive/references/).

## Local data -- `load_experiments/data/`

Default local inputs are:

- `load_experiments/data/processed/factual_2025_05_29_esm5.parquet`:
  processed aggregate queue input for the lifecycle experiment.
- `load_experiments/data/raw/2025_05_29_ESM5.parquet`: raw first-level
  depth/order-flow rows used only by archived diagnostics.

These data files are intentionally gitignored. See
[`load_experiments/data/README.md`](load_experiments/data/README.md) for the
expected local layout.

## Components and conventions

Reusable dataframe, lifecycle, impact, and accounting helpers live in `core/`;
the canonical runnable workflow lives in `load_experiments/`.

For the current file map, config sections, data schema, and lifecycle outputs,
see [`COMPONENTS.md`](COMPONENTS.md).
