# Impact Cost Components

This directory is an experiment layer around the Rust/native `simproj`
bindings. It adapts empirical order-flow rows into the formats needed to run
anchored queue and passive impact-cost experiments.

## Directory Map

| Path | Role | Keep in git? |
| --- | --- | --- |
| `README.md` | Conceptual notes for the anchored queue experiment. | Yes |
| `COMPONENTS.md` | File-by-file guide to code, inputs, outputs, and accepted options. | Yes |
| `__init__.py` | Re-exports a small public helper surface from `core/`. | Yes |
| `core/` | Reusable Python adapters and dataframe utilities. | Yes |
| `pipelines/` | Runnable experiment entry points. | Yes |
| `data/raw/` | Local raw order-flow/depth parquet files. | No, except README |
| `data/processed/` | Local processed/aggregated parquet files. | No, except README |
| `runs/` | Generated CSV/PNG/JSON outputs. | No, except README |
| `references/` | Local PDFs and bulky reference material. | No, except README |
| `archive/prototypes/` | Standalone exploratory code not imported by active pipelines. | Yes, if intentionally preserved |

## Data Schemas

Raw first-level depth input:

- Default path: `experiments/impact_cost/data/raw/2025_05_29_ESM5.parquet`
- Required columns for active pipelines: `ts`, `order_type`, `side`, `qty`, plus
  `a_1` or `b_1`
- Current file also contains depth columns `b_1..b_10` and `a_1..a_10`
- `order_type` values are expected to normalize to `limit`, `cancel`, or
  `market`
- `side` values are `A` or `B`
- `qty` is interpreted as nonnegative row size

Processed depth/order-flow input:

- Default path:
  `experiments/impact_cost/data/processed/factual_2025_05_29_esm5.parquet`
- Required columns: `ts`, `order_type`, `side`, `qty`, `q_b`, `q_a`
- Used by anchored queue simulation and impact-cost pipelines

Side/queue convention:

- Passive buy/bid defaults: `raw_side="A"`, `queue_col="q_b"`,
  `market_side="B"`; raw first-level execution maps to `b_1`
- Passive sell/ask alternative: `raw_side="B"`, `queue_col="q_a"`,
  `market_side="A"`; raw first-level execution maps to `a_1`
- `raw_side` is the posting side used for limit/cancel queue updates.
  `market_side` is the consuming market side used for fills and impact event
  times.
- Native single-queue impact is a queue-displacement contribution. The Python
  impact pipelines multiply bid-queue (`q_b`/`b_1`) contributions by `-1` to
  report signed price impact, so passive buy impact is positive with the
  default negative queue slope (`propagator_gamma` for
  reduced-form/tail-propagator, `c_kappa` for structural).

Reduced-form price-impact convention:

- Active default: `impact_model="reduced_form"`, implementing the Remark 2.4
  observable price approximation from the local paper draft.
- Direct image defaults: `propagator_kappa=0.00895780`,
  `propagator_gamma=-0.00001713`,
  `propagator_weights=(-0.00102289, 0.00084759, 0.00161378, 0.00031951)`,
  `propagator_beta=(10.0, 1.0, 0.1, 0.01)`.
- Notation: `propagator_kappa` is the constant propagator level `kappa_s`;
  `propagator_gamma` is the reduced-form queue slope `kappa_1`.
- For passive limits, signed market-order flow is shared across factual and
  no-us paths, so the pure propagator part cancels; the implemented passive
  contribution is `kappa_1 * int signed_queue_diff dN`.
- `impact_model="tail_propagator"` uses the same fitted propagator as input to
  the Rust-style passive continuation formula. It tracks exponential states at
  market events with `eta_i = beta_i w_i / (kappa_s * (beta_i + C_lambda))`,
  where `C_lambda=b_c-b_l`; `propagator_tail_zeta` controls the baseline
  continuation term and defaults to `0.0`. `propagator_tail` is kept as a
  compatibility alias. Structural-only `hawkes_*` and `c_kappa` fields do not
  enter this branch.
- The older Hawkes/tail-intensity path is still available with
  `impact_model="structural"` and the `hawkes_*` / `c_kappa` parameters.

Affine intensity slope convention:

- The active impact-cost pipelines use the smaller raw-slope convention by
  default: `b_l=-0.000097`, `b_c=0.0000989`.
- The previous x100 diagnostic convention is
  `b_l=-0.0097`, `b_c=0.00989`; keep it as an explicit scenario choice rather
  than the default.
- Intercepts are unchanged where anchored queue simulation is used:
  `a_l=184.1372`, `a_c=184.0456`.

Anchored native quantity convention:

- The general simulation library still represents events as unit-atomic, but
  the native impact-cost anchored simulator accepts empirical row sizes
  directly.
- A factual row with `qty=n` stays one native anchored row. For limit/cancel
  rows, intensities are evaluated at the pre-row anchored state:
  `lambda_bar = lambda(bar_q_t-)` and
  `lambda_q = lambda(bar_q_t- + dq_t-)`.
- The native simulator draws `n` accept/reject uniforms against those frozen
  intensities, applies one sized update to `bar_q` with quantity `n`, and
  applies one sized update to the simulated no-us queue with the accepted
  quantity.
- Consuming-side market rows are common background rows and are applied with
  full quantity to both factual and no-us paths.
- Empirical queue jumps that are not explained by simple unit replay stay in
  the exogenous anchored path `bar_q`; they are not converted into model
  events.

## Core Modules

The core layer is grouped by the passive impact-cost goal:

- empirical queue anchoring: `anchored_simulator.py`;
- execution labels and fill tracking: `level_execution.py`, `cost_utils.py`,
  `latency_filters.py`;
- reduced-form post/fill/cancel execution policy: `passive_lifecycle.py`;
- image-calibrated Remark 2.4 pricing: `reduced_form_impact.py`;
- market-time impact and execution-time cost jumps: `passive_impact.py`.

### `core/cost_utils.py`

Reusable dataframe utilities for passive execution and cost accounting.

Public pieces:

- `event_seconds(df, ts_col="ts", origin=None)`: converts timestamps to seconds
  from `origin`; accepts numeric seconds or datetime-like timestamps.
- `limit_event_positions(...)`: returns row positions of matching limit rows,
  optionally filtered by side and level.
- `flag_passive_limits(...)`: flags passive limit rows using one of
  `first_every`, `random_fraction`, or `indices`.
- `expand_event_times_by_dim(...)`: expands sized rows into repeated unit-event
  timestamps keyed by native event dimension.
- `regroup_event_times_by_dim(...)`: groups repeated unit events back into
  `(time, dim, qty)` rows.
- `track_passive_fills(..., side, market_side=None, ...)`: pure Python passive
  fill tracker for flagged limit orders. `side` is the posting side and
  `market_side` is the consuming market side. Supports cancellation policies
  `top`, `below`/`position`, and `probabilistic_top`.
- `cost_from_fills(...)`: computes total passive cost from fill times and a
  left-limit impact path.

Important accepted columns:

- Event tables: `ts`, `order_type`, `side`, `qty`
- Optional queue column for fill tracking: usually `q_a`, `q_b`, `a_1`, or
  `b_1`
- Optional level column: any column where `target_level` identifies the target
  execution level

### `core/anchored_simulator.py`

Adapter around native anchored queue simulation. It treats the empirical queue
path as `bar_q` and simulates only the displacement `dq = q - bar_q`.

Public pieces:

- `build_anchored_events(...)`: converts a dataframe window into native-ready
  rows with `time`, `dim`, `qty`, `bar_q_pre`, `bar_q_post`, and
  `is_passive_ours`.
- `event_dims_for_side(...)`: maps matching side/type rows to dimensions
  `0=limit`, `1=cancel`, `2=market`; non-modeled rows get `-1`.
- `select_passive_limit_flags(...)`: native policy helper for `none`,
  `first_every`, `random_fraction`, or `indices`.
- `simulate_anchored_queue_paths(...)`: runs the native anchored queue simulator
  and returns `AnchoredSimulationResult`.

Inputs:

- A dataframe window with `ts`, `order_type`, `side`, `qty`, and `queue_col`
- `passive_l_flags`: one boolean per row
- Affine intensity parameters `a_l`, `b_l`, `a_c`, `b_c`
- A time grid in seconds

Outputs:

- Empirical queue path, mechanical no-us queue path, simulated queue paths,
  simulated offsets, anchored event table, and regrouped simulated events

### `core/level_execution.py`

Helpers for converting raw first-level snapshot files into executable level-1
events.

Public pieces:

- `q1_column_for_side(raw_side, queue_col=None)`: maps the experiment queue
  convention to `a_1` or `b_1`.
- `market_side_for_queue(raw_side, queue_col=None, market_side=None)`: resolves
  the consuming market side for the selected queue.
- `price_sign_for_queue(queue_col)`: returns `+1` for ask queues and `-1` for
  bid queues when converting single-queue contribution to signed price impact.
- `first_level_execution_events_from_snapshots(...)`: infers first-level
  limit/cancel/market sizes from post-event q1 snapshots.
- `load_first_level_execution_window(...)`: loads a raw parquet window and
  returns inferred first-level execution rows.

### `core/latency_filters.py`

Filters latency-grid outputs before impact-cost simulation.

Public pieces:

- `LatencyFilterConfig`: controls max latency, completion requirement,
  selection mode, minimum selected orders, and required slots.
- `select_latency_orders(orders, cfg)`: returns selected orders and a selected
  window summary.

Accepted `selection_mode` values:

- `orders`: keep every order satisfying the order-level filter
- `window_any`: keep windows with at least one selected order
- `window_at_least`: keep windows with at least `min_orders` selected orders
- `window_all`: keep windows where all required slots were selected

### `core/reduced_form_impact.py`

Image-calibrated reduced-form price-impact coefficients and helpers for
Remark 2.4.

Public pieces:

- `ReducedFormPropagator`: fitted direct price kernel
  `kappa + sum_i w_i exp(-beta_i t)`.
- `passive_reduced_form_impact_from_queue_samples(...)`: passive queue
  correction when signed market-order flow is shared across factual and no-us
  paths.
- `propagator_impact_from_events(...)`: direct propagator response for extra
  signed-flow events, such as aggressive metaorder fills.

### `core/passive_impact.py`

Final passive accounting layer from queue samples to impact/cost time series.

Public pieces:

- `PassiveImpactModelConfig`: chooses `reduced_form`, `tail_propagator`, or
  `structural` impact.
- `passive_impact_path_from_queue_samples(...)`: returns market-time impact
  aligned to consuming market events.
- `execution_cost_jump_series(...)`: samples the impact path at fill times and
  returns cost jumps plus cumulative running cost.
- `passive_cost_from_fills(...)`: returns total passive execution cost from
  fill labels and a market-time impact path.

### `core/passive_lifecycle.py`

Synthetic passive order lifecycle policy, independent from market data and
price-impact math.

Public pieces:

- `PassiveLifecycleConfig`: controls cycle count, orders per cycle, posting
  spacing, fill-count rule, fill-time rule, cancel timing, and repost delay.
- `generate_passive_lifecycle(...)`: returns `orders`, `fills`, `cancels`,
  `events`, and `cycle_summary` frames for one sampled policy path.
- `active_displacement_at_times(events, times)`: evaluates the active displayed
  own quantity curve implied by post/fill/cancel events.

Default policy:

- `K ~ Binomial(orders_per_cycle, fill_probability)` with
  `fill_probability=1/7`
- clustered exponential fill timing after the posting phase
- unfilled orders cancel after `min_resting_seconds` plus optional jitter
- the next cycle starts after `repost_delay_seconds`

## Pipelines

Run pipelines from the repository root with:

```bash
python -m experiments.impact_cost.pipelines.<module_name> --help
```

### `pipelines/lifecycle_passive_cost_pipeline.py`

Looped passive lifecycle cost experiment. It uses empirical consuming-side
market-event times and factual aggregate queues, but generates our own
post/fill/cancel lifecycle in reduced form.

Core rule:

```text
q_no_us(t) = max(q_factual(t) - active_own_qty(t), 0)
```

Fills create cost jumps; cancels only remove active displayed quantity for
future impact.

Main options:

- `--n-cycles`, `--orders-per-cycle`, `--posting-spacing-seconds`
- `--fill-count-model {binomial,fixed}`
- `--fill-probability`, `--fixed-filled-orders`
- `--fill-time-model {clustered_exponential,independent_exponential}`
- `--fill-wait-mean-seconds`, `--fill-gap-mean-seconds`
- `--min-resting-seconds`, `--cancel-delay-seconds`,
  `--cancel-jitter-seconds`, `--repost-delay-seconds`
- `--warmup-seconds`: pre-posting market history used to warm the
  propagator-tail states with zero own displacement
- `--n-policy-paths`, `--horizon-seconds`, `--impact-model`

Key outputs:

- `policy_orders.csv`, `policy_fills.csv`, `policy_cancels.csv`,
  `policy_events.csv`, `policy_cycle_summary.csv`
- `active_quantity_path_samples.csv` and summary
- `price_impact_path_samples.csv` and summary
- `impact_cost_fill_jumps.csv`
- `impact_cost_path_samples.csv`, `impact_cost_path_summary.csv`, and
  `impact_cost_path_summary_by_fill_count.csv`
- `lifecycle_impact_cost_paths.png`

### `pipelines/queue_pipeline.py`

Anchored conditional queue-path simulation over the processed depth input.

Default input:

- `experiments/impact_cost/data/processed/factual_2025_05_29_esm5.parquet`

Main options:

- `--parquet-path`: processed parquet or CSV with `ts/order_type/side/qty/q_b/q_a`
- `--output-dir`: run output directory
- `--horizon-seconds`: window length from `--start-time`
- `--start-time`: optional window start; defaults to first timestamp
- `--raw-side`: passive posting side modeled by the simulator
- `--queue-col`: empirical queue column, usually `q_a` or `q_b`
- `--market-side`: consuming market side; defaults to the passive buy/bid
  convention in active pipelines
- `--selection-policy`: `first_every`, `random_fraction`, `indices`, or `none`
- `--every-seconds`: spacing for `first_every`
- `--selection-fraction`: fraction for `random_fraction`
- `--selection-indices`: comma-separated indices for `indices`
- `--selection-stop-seconds`: optional cutoff for selecting passive orders
- `--n-simulations`, `--n-grid`, `--seed`
- `--a-l`, `--b-l`, `--a-c`, `--b-c`: affine intensity parameters
- `--require-replay-match`, `--replay-tolerance`: optional diagnostic guard

Outputs include:

- `conditional_queue_paths.png`
- `conditional_queue_offsets.png`
- `replay_divergence.png`
- `anchor_report.csv`
- `replay_consistency.csv`
- selected passive order rows and simulated queue arrays/events
- `queue_pipeline_config.json`

### `pipelines/execution_latency_grid.py`

Measures passive fill latency on raw first-level depth, minute by minute.

Default input:

- `experiments/impact_cost/data/raw/2025_05_29_ESM5.parquet`

Main options:

- `--raw-level-path`: raw parquet with `ts/order_type/side/qty/a_1/b_1`
- `--output-dir`
- `--start-time`, `--end-time`
- `--minute-seconds`: spacing between windows
- `--n-orders`: number of posting slots per window
- `--order-spacing-seconds`: spacing between posting slots
- `--tracking-horizon-seconds`: fill-tracking horizon
- `--raw-side`, `--queue-col`, `--market-side`
- `--cancellation-policy`: `top`, `below`/`position`, or `probabilistic_top`
- `--theta`: probability for `probabilistic_top`
- `--cap-position-by-queue-post`: cap tracked position at observed q1
- `--seed`

Outputs include:

- `passive_execution_latencies_by_minute.csv`
- `passive_execution_fills_by_minute.csv`
- `passive_execution_minute_summary.csv`
- `passive_execution_latency_by_minute.png`
- `passive_execution_latency_histogram.png`
- `execution_latency_grid_config.json`

### `pipelines/impact_cost_pipeline.py`

Filters latency-grid orders and estimates passive impact cost by re-simulating
anchored no-us queues for selected windows.

Inputs:

- Processed depth input via `--aggregated-path`
- Latency-grid order CSV via `--latency-orders-path`
- Latency-grid fill CSV via `--latency-fills-path`

Main options:

- `--max-latency-seconds`: order-level latency filter; use `None` from Python
  when constructing `ImpactCostPipelineConfig` to disable this filter
- `--selection-mode`: `orders`, `window_any`, `window_at_least`, or `window_all`
- `--min-orders`
- `--required-slots`: comma-separated zero-based order slots
- `--max-windows`
- `--horizon-seconds`
- `--n-simulations`
- `--raw-side`, `--queue-col`, `--market-side`, `--seed`
- `--a-l`, `--b-l`, `--a-c`, `--b-c`: affine queue parameters
- `--impact-model`: `reduced_form` by default; `tail_propagator` uses the
  native fitted-propagator continuation model; `structural` uses the older
  Hawkes/tail-intensity passive flow model
- `--propagator-kappa`, `--propagator-gamma`, `--propagator-weights`,
  `--propagator-beta`: fitted reduced-form price-propagator coefficients from
  the reference image
- `--c-kappa`, `--hawkes-mu`, `--hawkes-alpha`, `--hawkes-beta`: structural
  passive flow impact parameters, only used with `--impact-model structural`

Outputs include:

- `selected_latency_orders.csv`
- `selected_latency_windows.csv`
- `selected_latency_fills.csv`
- `impact_cost_samples.csv`
- `impact_cost_fill_contributions.csv`
- `impact_cost_window_summary.csv`
- `impact_cost_distribution.png`
- `impact_cost_config.json`

### `pipelines/scheduled_passive_cost_pipeline.py`

Validation pipeline for scheduled passive posting/fill logic. It selects the
first factual passive limit row in each posting bucket, treats those rows as
our passive orders, simulates anchored no-us queues, and applies synthetic fill
times to produce cumulative impact-cost step paths.

Default schedule:

- post phase: 10 buckets of 10ms each, i.e. a 100ms posting window
- fill phase: first fill at 150ms with deterministic exponential-quantile
  timing; default half-life is 10ms
- queue parameters: unscaled slopes `b_l=-0.000097`, `b_c=0.0000989`

Main options:

- `--episode-spacing-seconds`: spacing between candidate window anchors
- `--max-episodes`, `--randomize-episodes`
- `--posting-spacing-seconds`, `--n-posting-slots`
- `--fill-start-seconds`, `--fill-schedule`, `--fill-spacing-seconds`,
  `--fill-half-life-seconds`, `--n-filled-orders`
- `--allow-incomplete-posting-grid`: keep windows with fewer populated
  posting buckets and stratify summaries by `n_filled_orders`
- `--horizon-seconds`, `--output-step-seconds`, `--n-simulations`
- `--raw-side`, `--queue-col`, `--market-side`
- `--a-l`, `--b-l`, `--a-c`, `--b-c`
- `--impact-model` and the reduced-form/structural impact parameters shared
  with the other impact pipelines

Outputs include:

- `selected_orders.csv`
- `synthetic_fills.csv`
- `impact_cost_fill_jumps.csv`: one row per fill, simulation, and episode
- `impact_cost_path_samples.csv`: cumulative step path on a shared grid
- `impact_cost_path_summary.csv`: mean and quantiles by `n_filled_orders` and
  aligned time
- `price_impact_path_samples.csv`: underlying reduced-form price-impact path
  on the same shared grid
- `price_impact_path_summary.csv`: mean and 5-95% band for the background
  price-impact curve
- `episode_summary.csv`
- `impact_cost_paths.png`: zoomed execution-cluster view with shadowed paths
  plus mean and 5-95% band, with mean price impact in the background
- `scheduled_passive_cost_config.json`

### `pipelines/impact_series_pipeline.py`

Randomly samples factual passive limit-addition sequences and outputs aligned
impact time series. This is an order-of-magnitude diagnostic: selected limit
rows are treated as synthetic "our orders," but no real-fill condition is
imposed. Running the full pipeline requires the compiled native `simproj`
extension because the anchored no-us simulation lives in Rust.

Default input:

- `experiments/impact_cost/data/processed/factual_2025_05_29_esm5.parquet`

Main options:

- `--aggregated-path`: processed parquet or CSV with `ts/order_type/side/qty/q_b/q_a`
- `--output-dir`
- `--n-orders-per-episode`: number of selected limit rows per episode, e.g. `7`
- `--n-episodes`: number of random factual sequences to sample
- `--post-span-seconds`: maximum time between first and last selected limit row
- `--horizon-seconds`: impact curve horizon after the first selected limit
- `--output-step-seconds`: aligned output grid spacing
- `--n-simulations`: no-us baseline replicas per episode
- `--raw-side`, `--queue-col`, `--market-side`
- `--start-time`, `--end-time`
- `--min-market-events`: skip/silence episodes with too few market events
- `--seed`
- `--a-l`, `--b-l`, `--a-c`, `--b-c`: affine queue parameters
- `--impact-model`: `reduced_form` by default; `tail_propagator` uses the
  native fitted-propagator continuation model; `structural` uses the older
  Hawkes/tail-intensity passive flow model
- `--propagator-kappa`, `--propagator-gamma`, `--propagator-weights`,
  `--propagator-beta`: fitted reduced-form price-propagator coefficients from
  the reference image
- `--c-kappa`, `--hawkes-mu`, `--hawkes-alpha`, `--hawkes-beta`: structural
  passive flow impact parameters, only used with `--impact-model structural`

Outputs include:

- `selected_episodes.csv`
- `selected_episode_orders.csv`
- `impact_series_samples.csv`: one row per episode, simulation, and aligned
  time point
- `impact_series_summary.csv`: mean, standard deviation, and quantiles by
  aligned time
- `episode_summary.csv`
- `impact_series_summary.png`: all sampled paths in faint grey, with mean and
  5-95% band overlaid
- `impact_series_config.json`

### `pipelines/running_cost_diagnostics.py`

Plots per-order queue priority and running cost for selected impact-cost
orders.

Inputs:

- Raw depth parquet via `--raw-level-path`
- Impact-cost output directory via `--cost-output-dir`

Main options:

- `--output-dir`
- `--raw-side`, `--queue-col`, `--market-side`
- `--cancellation-policy`
- `--cap-position-by-queue-post`
- `--max-plots`
- `--order-by`: `window_id`, `latency`, or `abs_cost`

Outputs include:

- `manifest.csv`
- one `window_<id>_order_<id>.png` plot per selected order
- `running_cost_diagnostics_config.json`

### `pipelines/execution_overlay.py`

Visual diagnostic for one raw first-level window. It overlays passive postings
and fills on the raw q1 path.

Main options:

- `--raw-level-path`
- `--output-dir`
- `--image-dir`
- `--start-time`
- `--horizon-seconds`
- `--tracking-horizon-seconds`
- `--every-seconds`
- `--selection-stop-seconds`
- `--raw-side`, `--queue-col`, `--market-side`
- `--cancellation-policy`, `--theta`, `--seed`

Outputs include:

- `passive_execution_times.csv`
- `passive_execution_times_until_filled.csv`
- `passive_execution_overlay.png`
- `passive_execution_overlay_until_filled.png`
- `execution_overlay_config.json`

## Archive

`archive/prototypes/metaorder_detection.py` is not imported by active code. It
contains exploratory helpers for assigning trade rows to synthetic metaorders.
Keep it separate from active pipelines unless it becomes part of a tested flow.

## Suggested Build Order

1. Validate anchoring with `queue_pipeline` when changing queue conventions or
   data inputs.
2. Produce execution labels with `execution_latency_grid`, or provide an
   equivalent metaorder/fill table with execution-time labels.
3. Run `impact_cost_pipeline` to simulate no-us queues, compute reduced-form
   impact paths, and output execution-time cost jumps.
4. Run `running_cost_diagnostics` on the impact-cost output when you need
   per-order visual checks.
5. Run `impact_series_pipeline` for unconditional/random passive limit-sequence
   impact curves without real-fill conditioning.
6. Use `execution_overlay` as a lower-level visual diagnostic for fill-tracking
   assumptions.
