# Point Process Simulation

Conditional path simulation for a **one-dimensional point process** after
inserting one or more exogenous events. The baseline path is used as the
conditioning trajectory; the shocked paths are sampled under the same model
with the perturbation injected as external flow.

## Custom experiment — `custom_experiment/`

Edit the configs and `process_type` in
[`custom_experiment/main.py`](custom_experiment/main.py) and run:

```bash
python experiments/point_process_simulation/custom_experiment/main.py
```

Set `process="hawkes"` for the multi-exponential Hawkes process, or
`process="affine"` for the counting process
$\lambda(N_t) = b + aN_t$.

Knobs: `time_horizon`, `n_simulations`, `process`, the Hawkes parameters
`mu`/`alpha`/`beta`, `stationary`, the affine parameters `a`/`b`,
`perturbation_time` (a scalar time or an explicit list/array of forced event
times), `shared_acceptance`, and `seed`.

Outputs (`baseline_times.npy`, `perturbation_times.npy`,
`perturbed_times.npy`, `perturbed_lengths.npy`, `baseline_count.npy`,
`time_horizon.npy`, `process_kind.npy`, `affine_a.npy`, `affine_b.npy`) land
in `custom_experiment/output/affine/` or `custom_experiment/output/hawkes/`
(gitignored).

Generate the factual-vs-shocked counting-process plot with:

```bash
python experiments/point_process_simulation/custom_experiment/plot_utils.py
```

By default the plot script scans both output subdirectories and writes separate
images under `custom_experiment/images/` for every process with available
outputs.

`perturbed_times.npy` is padded with `NaN`; use `perturbed_lengths.npy` for the
valid event count in each simulated path.
