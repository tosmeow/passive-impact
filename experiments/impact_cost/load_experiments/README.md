# Impact Cost Lifecycle Experiment

Canonical load experiment for passive impact cost. Edit `config.toml`, run the
lifecycle experiment, and regenerate the display figure with `plot_utils.py`.

## Run

```bash
python -m experiments.impact_cost.load_experiments.lifecycle_passive_cost \
  --config experiments/impact_cost/load_experiments/config.toml
```

Run this from the repository root. Do not use the old
`experiments.impact_cost.pipelines...` module or `experiments/impact_cost/runs`
config paths; those are archived development paths.

If `simproj` is not installed in your active environment, the runner tries the
repo-local native package under `code/python/`. Avoid putting a temporary wheel
directory such as `/private/tmp/simproj-wheel-extracted-current` on
`PYTHONPATH` unless it was built for the same Python version as your shell. If
you see an `_native` import error from `/private/tmp`, run `unset PYTHONPATH`
and rerun the command above.

The runner is fixed to the `tail_propagator` impact model. Runtime overrides are
limited to output paths, `max_episodes`, `n_policy_paths`, `seed`, and episode
randomization; model and lifecycle parameters should be changed in
`config.toml`.

The checked-in config defaults to a small showcase run (`max_episodes = 20`,
`n_policy_paths = 10`) so figures can be regenerated quickly. Increase those
two values in `config.toml` for larger sweeps.

The lifecycle keeps randomized fill/execution times, but resolves generated
post/cancel displayed quantity to the next observed limit/cancel row quantities
on the selected side. There is no synthetic post/cancel timestamp mode in the
canonical runner.

To retarget the propagator to a chosen implied Hawkes norm while preserving the
shape stored in `base_config.toml`, run:

```bash
python -m experiments.impact_cost.load_experiments.set_propagator_norm 0.95
```

This rewrites only the `propagator_weights` line in `config.toml`.

## Display

```bash
python -m experiments.impact_cost.load_experiments.plot_utils
```

CSV/JSON outputs are written under `data/lifecycle_passive_cost/`; the
canonical PNG is written to `images/lifecycle_impact_cost_paths.png`.
