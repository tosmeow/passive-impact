"""Run a parameter-tweaked aggressive impact experiment."""
from pathlib import Path

import numpy as np

from simproj import agressive_impact as ai

# ──────────────── CONFIG ────────────────
config = ai.AggressiveImpactConfig(
    time_horizon=100.0,
    n_simulations=500,
    initial_queue_size=200,
    counterfactual=False,         # False: with us | True: without us
    model="propagator",          # "propagator" | "hybrid"
    bar_kappa=None,              # required when model == "hybrid"
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    # kappa defaults to c1 * sqrt(log(e^{-c2 q} + 1)); override if needed:
    # kappa=lambda q: 800.0 * np.sqrt(q),
    metaorder=200,
    metaorder_window=(1.0, 75.0),
    seed=42,
)
# ────────────────────────────────────────

if __name__ == "__main__":
    result = ai.run(config)
    direction = "without_us" if config.counterfactual else "with_us"
    output_dir = Path(__file__).resolve().parent / "output" / direction
    ai.save(result, str(output_dir))
    print(f"Done. Outputs in {output_dir}")
