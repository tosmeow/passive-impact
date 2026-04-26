import numpy as np
import simproj


def test_affine_queue_smoke():
    hawkes = simproj.MultiExponentialHawkes.with_stationary_state(
        mu=1.0, alpha=[0.065, 0.2, 0.325, 0.65], beta=[0.15, 0.6, 2.5, 10.0],
    )
    market_orders = simproj.simulate_hawkes_as_market_orders(hawkes, t_max=10.0, seed=42)
    assert len(market_orders) > 0

    process = simproj.AffineQueueProcess.new_queue(
        q0=200.0, a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    )
    q_events = simproj.simulate_with_externals(process, 10.0, market_orders, seed=42)
    full = simproj.merge_events(q_events, market_orders)
    samples = simproj.sample_queue_at_times(
        full, initial_q=200,
        times=np.linspace(0.0, 10.0, 11).astype(np.float64),
    )
    assert samples.shape == (11,)
    assert samples.dtype == np.uint32


def test_create_meta_orders_explicit_times():
    times = np.array([1.0, 2.0, 4.0, 8.0])
    meta = simproj.create_meta_orders_from_times(times, target_dim=2, total_dims=3)
    assert len(meta) == 4
    assert np.allclose(meta.times(), times)
    assert np.all(meta.dims() == 2)


def test_c_lambda_helper():
    assert simproj.AffineQueueProcess.c_lambda(b_l=-0.275, b_c=0.125) == 0.4
