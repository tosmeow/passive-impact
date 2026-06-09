# Load Experiment Data

Local inputs and generated outputs for the canonical lifecycle experiment live
here.

- `processed/factual_2025_05_29_esm5.parquet`: processed aggregate queue input
  used by `config.toml`.
- `raw/2025_05_29_ESM5.parquet`: raw first-level depth/order-flow input kept
  for archived diagnostics.
- `lifecycle_passive_cost/`: generated CSV/JSON outputs from the canonical
  lifecycle run.

Parquet inputs and generated outputs are intentionally ignored by git.
