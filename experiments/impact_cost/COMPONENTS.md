# Impact Cost Components

This document describes the current lifecycle impact-cost experiment. It does
not document archived diagnostics or earlier prototypes.

## Live Layout

| Path | Role |
| --- | --- |
| `README.md` | Short overview and run command. |
| `COMPONENTS.md` | Current workflow map. |
| `core/` | Reusable dataframe, lifecycle, impact, and accounting helpers. |
| `load_experiments/config.toml` | Main file to edit before running the experiment. |
| `load_experiments/lifecycle_passive_cost.py` | Canonical lifecycle runner. |
| `load_experiments/plot_utils.py` | Regenerates the canonical figure from saved outputs. |
| `load_experiments/data/` | Local parquet inputs and generated CSV/JSON outputs. |
| `load_experiments/images/` | Generated PDF figures. |

The `archive/` folder is retained for development history. It is not part of
the current lifecycle workflow.

## Input Data

The canonical run reads one processed aggregate queue file by default:

```text
experiments/impact_cost/load_experiments/data/processed/factual_2025_05_29_esm5.parquet
```

Required columns:

- `ts`: event timestamp
- `order_type`: event type
- `side`: event side
- `qty`: event size
- `q_b`: factual bid queue
- `q_a`: factual ask queue

Local parquet files are intentionally ignored by git.

## Config

Edit `load_experiments/config.toml`.

The config has five sections:

- `[paths]`: input parquet, CSV/JSON output directory, image directory
- `[episodes]`: episode sampling, horizon, grid step, warmup, seed
- `[queue]`: queue side conventions
- `[lifecycle]`: random lifecycle rules and empirical post/cancel row resolution
- `[impact]`: fixed tail-propagator coefficients

The current experiment is fixed to `tail_propagator`. There is no
`impact_model` switch in the config.

## Workflow

Run:

```bash
python -m experiments.impact_cost.load_experiments.lifecycle_passive_cost \
  --config experiments/impact_cost/load_experiments/config.toml
```

The runner does the following:

1. Loads the processed aggregate queue.
2. Samples candidate lifecycle episodes from empirical market-event windows.
3. Generates random passive lifecycle intentions.
4. Resolves intended post/cancel displayed quantity onto the next observed
   limit/cancel row quantities on the selected side. Fill/execution times remain
   the random lifecycle times; they are not inferred from observed market
   events.
5. Converts active displayed quantity into a no-us queue:

```text
q_no_us(t) = max(q_with_us(t) - active_own_qty(t), 0)
```

6. Computes the tail-propagator passive price-impact path.
7. Samples the left-limit impact at fill times.
8. Accumulates running execution cost:

```text
C_t = sum_{fills tau_j <= t} qty_j * Delta P_{tau_j-}
```

Cancels remove active displayed quantity for future impact. They do not create
cost jumps.

## Side Conventions

Default passive buy/bid run:

- `raw_side = "A"`
- `queue_col = "q_b"`
- `market_side = "B"`

Passive sell/ask runs use:

- `raw_side = "B"`
- `queue_col = "q_a"`
- `market_side = "A"`

Bid-queue impacts are signed negative; ask-queue impacts are signed positive.

## Current Core Modules

- `core/experiment_utils.py`: load aggregate data, choose windows, sample paths
- `core/passive_lifecycle.py`: generate post/fill/cancel lifecycle events
- `core/empirical_lifecycle.py`: resolve lifecycle post/cancel intentions onto observed row quantities
- `core/passive_impact.py`: bridge queue paths to impact and cost jumps
- `core/reduced_form_impact.py`: tail-propagator impact implementation
- `core/cost_utils.py`: event-time and left-limit accounting utilities
- `core/level_execution.py`: queue-side mapping helpers

The lifecycle runner uses the native `simproj` tail-propagator helper when the
compiled extension is available, and otherwise falls back to the Python
implementation.

## Outputs

CSV/JSON outputs are written under:

```text
load_experiments/data/lifecycle_passive_cost/
```

Main output files:

- `episode_summary.csv`
- `policy_path_summary.csv`
- `policy_orders.csv`
- `policy_fills.csv`
- `policy_cancels.csv`
- `policy_events.csv`
- `policy_cycle_summary.csv`
- `policy_unresolved_events.csv`
- `impact_cost_path_summary.csv`
- `price_impact_path_summary.csv`
- `active_quantity_path_summary.csv`
- `lifecycle_passive_cost_config.json`

PDF figures are written under:

```text
load_experiments/images/
```

Generated figure:

- `lifecycle_impact_cost_paths.pdf`

Regenerate figures from saved CSV outputs with:

```bash
python -m experiments.impact_cost.load_experiments.plot_utils
```
