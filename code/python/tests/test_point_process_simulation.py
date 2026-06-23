import numpy as np
import simproj
from simproj import point_process_simulation as pps


def test_conditional_hawkes_simulate_many_records_perturbation():
    hawkes = simproj.MultiExponentialHawkes.with_stationary_state(
        mu=1.0, alpha=[0.065, 0.2], beta=[0.15, 0.6],
    )
    baseline = simproj.simulate_hawkes_result(hawkes, t_max=5.0, seed=42)
    perturbation = simproj.create_meta_orders_from_times(
        np.array([1.0], dtype=np.float64), target_dim=0, total_dims=1,
    )

    ctx = simproj.ConditionalHawkesSimulationContext(
        hawkes,
        [list(baseline.times())],
        5.0,
        new_externals=perturbation,
    )

    paths = ctx.simulate_many(n_simulations=3, base_seed=100)
    assert len(paths) == 3
    for path in paths:
        assert np.any(np.isclose(path.times(), 1.0))
        assert np.all(path.dims() == 0)

    time_paths = ctx.simulate_many_times(n_simulations=3, base_seed=100)
    assert len(time_paths) == 3
    for times in time_paths:
        assert np.any(np.isclose(times, 1.0))


def test_conditional_affine_counting_simulate_many_records_perturbation():
    process = simproj.AffineCountingProcess(a=0.2, b=0.8)
    baseline = simproj.simulate_affine_counting_process(process, t_max=5.0, seed=42)
    perturbation = simproj.create_meta_orders_from_times(
        np.array([1.0], dtype=np.float64), target_dim=0, total_dims=1,
    )

    ctx = simproj.ConditionalAffineCountingSimulationContext(
        process,
        [list(baseline.times())],
        5.0,
        new_externals=perturbation,
    )

    paths = ctx.simulate_many(n_simulations=3, base_seed=100)
    assert len(paths) == 3
    for path in paths:
        assert np.any(np.isclose(path.times(), 1.0))
        assert np.all(path.dims() == 0)

    time_paths = ctx.simulate_many_times(n_simulations=3, base_seed=100)
    assert len(time_paths) == 3
    for times in time_paths:
        assert np.any(np.isclose(times, 1.0))


def test_point_process_facade_pads_ragged_paths():
    cfg = pps.PointProcessSimulationConfig(
        time_horizon=5.0,
        n_simulations=4,
        mu=1.0,
        alpha=[0.065, 0.2],
        beta=[0.15, 0.6],
        perturbation_time=1.0,
        seed=7,
    )

    out = pps.run(cfg)
    assert out["baseline_times"].ndim == 1
    assert np.allclose(out["perturbation_times"], [1.0])
    assert out["perturbed_times"].shape[0] == 4
    assert out["perturbed_lengths"].shape == (4,)
    assert np.all(out["perturbed_lengths"] > 0)
    assert np.allclose(out["time_horizon"], [5.0])


def test_point_process_facade_runs_affine_counting_process():
    cfg = pps.PointProcessSimulationConfig(
        process="affine",
        time_horizon=4.0,
        n_simulations=4,
        a=0.2,
        b=0.8,
        perturbation_time=1.0,
        seed=7,
    )

    out = pps.run(cfg)
    assert out["process_kind"][0] == "affine"
    assert np.allclose(out["affine_a"], [0.2])
    assert np.allclose(out["affine_b"], [0.8])
    assert np.allclose(out["perturbation_times"], [1.0])
    assert out["perturbed_times"].shape[0] == 4
    assert out["perturbed_lengths"].shape == (4,)
    assert np.all(out["perturbed_lengths"] > 0)
