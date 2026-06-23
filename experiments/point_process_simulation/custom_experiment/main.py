"""Run a conditional point-process perturbation experiment."""
from pathlib import Path

from simproj import point_process_simulation as pps

# ---------------- CONFIG ----------------

process_type = "affine"

config_affine = pps.PointProcessSimulationConfig(
    process="affine",
    time_horizon=1.0,
    n_simulations=500,
    a=1.0,
    b=10.0,
    # For Hawkes instead, set process="hawkes" and specify mu/alpha/beta.
    perturbation_time=0.1,
    # Use a list/array for multiple forced events, for example:
    # perturbation_time=[10.0, 10.5, 11.0],
    shared_acceptance=False,
    seed=42,
)

config_hawkes = pps.PointProcessSimulationConfig(
    process="hawkes",
    time_horizon=1.0,
    n_simulations=500,
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    stationary=True,
    perturbation_time=0.1,
    # Use a list/array for multiple forced events, for example:
    # perturbation_time=[10.0, 10.5, 11.0],
    shared_acceptance=False,
    seed=42,
)
# ----------------------------------------

if __name__ == "__main__":
    configs = {
        "affine": config_affine,
        "hawkes": config_hawkes,
    }
    if process_type not in configs:
        raise ValueError("process_type must be 'affine' or 'hawkes'")

    config = configs[process_type]
    result = pps.run(config)
    output_dir = Path(__file__).resolve().parent / "output" / process_type
    pps.save(result, str(output_dir))
    print(f"Done. Outputs in {output_dir}")
