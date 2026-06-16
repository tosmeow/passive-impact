"""Run a parameter-tweaked aggressive impact experiment."""
from pathlib import Path

from simproj import agressive_impact as ai

# ──────────────── CONFIG ────────────────
config = ai.AggressiveImpactConfig(
    time_horizon=100.0,
    n_simulations=500,
    initial_queue_size=200,
    counterfactual=False,         # False: with us | True: without us
    bar_kappa=0.01,
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    # kappa defaults to the hybrid linear correction kappa(q) = -0.001 * q.
    # kappa=lambda q: -0.001 * q,
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
