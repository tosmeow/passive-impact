# Point Process Simulation

Conditional path simulation for a **multi-exponential Hawkes process** after
inserting one or more exogenous events. The baseline path is used as the
conditioning trajectory; the shocked paths are sampled under the same Hawkes
kernel with the perturbation injected as external flow.

## Custom experiment — `custom_experiment/`

Edit the config block in [`custom_experiment/main.py`](custom_experiment/main.py)
and run:

```bash
python experiments/point_process_simulation/custom_experiment/main.py
```

Knobs: `time_horizon`, `n_simulations`, the Hawkes parameters
`mu`/`alpha`/`beta`, `stationary`, `perturbation_time` (a scalar time or an
explicit list/array of forced event times), `shared_acceptance`, and `seed`.

Outputs (`baseline_times.npy`, `perturbation_times.npy`,
`perturbed_times.npy`, `perturbed_lengths.npy`, `baseline_count.npy`,
`time_horizon.npy`) land in `custom_experiment/output/` (gitignored).

Generate the factual-vs-shocked counting-process plot with:

```bash
python experiments/point_process_simulation/custom_experiment/plot_utils.py
```

`perturbed_times.npy` is padded with `NaN`; use `perturbed_lengths.npy` for the
valid event count in each simulated path.
