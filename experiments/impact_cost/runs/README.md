# Runs

Generated outputs from impact-cost pipelines live here. This directory is
ignored by git except for this README.

Current convention:

- Keep the top-level `runs/` folder empty except for this README and
  `archive/` until a deliberately named final run is produced.
- Put provisional, smoke, diagnostic, or superseded outputs under `archive/`.
- Name future top-level folders according to the agreed experiment logic before
  launching the run.

`archive/2026-06-07_cleanup/` contains the previous latency-grid input,
provisional reduced-form unscaled queue run, older smoke, x100-slope,
impact-series, queue-diagnostic, and test outputs kept for reproducibility.
