"""Run a parameter-tweaked queue-only counterfactual simulation."""
from simproj import queue_simulation as qs

# ──────────────── CONFIG ────────────────
config = qs.QueueSimulationConfig(
    time_horizon=100.0,
    n_simulations=500,
    n_eval_times=1000,
    initial_queue_size=200,
    mode="single",
    counterfactual=False,         # False: with us | True: without us
    mu=1.0,
    alpha=[0.065, 0.2, 0.325, 0.65],
    beta=[0.15, 0.60, 2.5, 10.0],
    a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    metaorder=375,
    metaorder_window=(1.0, 80.0),
    seed=42,
)
# ────────────────────────────────────────

if __name__ == "__main__":
    result = qs.run(config)
    qs.save(result, "experiments/queue_simulation/custom_experiment/output/")
    print("Done. Outputs in experiments/queue_simulation/custom_experiment/output/")
