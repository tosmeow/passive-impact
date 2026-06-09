# Impact Cost Lifecycle Experiment

Canonical load experiment for passive impact cost. This directory mirrors the
older experiment layout: edit `config.toml`, run the lifecycle experiment, and
regenerate display figures with `plot_utils.py`.

## Run

```bash
python -m experiments.impact_cost.load_experiments.lifecycle_passive_cost \
  --config experiments/impact_cost/load_experiments/config.toml
```

The runner is fixed to the `tail_propagator` impact model. Runtime overrides are
limited to output paths, `max_episodes`, `n_policy_paths`, `seed`, and episode
randomization; model and lifecycle parameters should be changed in
`config.toml`.

The checked-in config defaults to a small showcase run (`max_episodes = 10`,
`n_policy_paths = 10`) so figures can be regenerated quickly. Increase those
two values in `config.toml` for larger sweeps.

## Display

```bash
python -m experiments.impact_cost.load_experiments.plot_utils
```

CSV/JSON outputs are written under `data/lifecycle_passive_cost/`; PNG figures
are written under `images/`.
