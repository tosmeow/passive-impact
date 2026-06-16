"""Run a parameter-tweaked passive impact experiment."""
from pathlib import Path

from simproj import passive_impact as pi

# ──────────────── CONFIG ────────────────
config = pi.PassiveImpactConfig(
    time_horizon=100.0,
    n_simulations=500,
    initial_queue_size=200,
    mode="single",                # "single" | "double"
    counterfactual=False,         # False: with us | True: without us
    side="ask",                   # "ask" keeps sign; "bid" flips impact sign
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    c_kappa_effective=-0.00001713,
    # Metaorder accepts:
    #   int N         → N evenly-spaced orders inside metaorder_window
    #   list/ndarray  → explicit list of arrival times (window ignored)
    metaorder=375,
    metaorder_window=(1.0, 80.0),
    # Example explicit list (commented out):
    # metaorder=[1.0, 2.5, 4.0, 7.0, 12.0, 30.0, 60.0],
    seed=42,
)
# ────────────────────────────────────────

if __name__ == "__main__":
    result = pi.run(config)
    direction = "without_us" if config.counterfactual else "with_us"
    output_dir = Path(__file__).resolve().parent / "output" / direction
    pi.save(result, str(output_dir))
    print(f"Done. Outputs in {output_dir}")
