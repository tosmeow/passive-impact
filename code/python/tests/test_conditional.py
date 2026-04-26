import numpy as np
import simproj


def test_conditional_simulate_queue_at_times():
    hawkes = simproj.MultiExponentialHawkes.with_stationary_state(
        mu=1.0, alpha=[0.065, 0.2, 0.325, 0.65], beta=[0.15, 0.6, 2.5, 10.0],
    )
    market_orders = simproj.simulate_hawkes_as_market_orders(hawkes, t_max=10.0, seed=42)

    process = simproj.AffineQueueProcess.new_queue(
        q0=200.0, a_l=100.0, b_l=-0.275, a_c=2.0, b_c=0.125,
    )
    q_events = simproj.simulate_with_externals(process, 10.0, market_orders, seed=42)
    cond_by_dim = simproj.extract_events_by_dim(q_events, total_dims=3, exclude_dim=2)

    meta = simproj.create_meta_orders(n=10, t_start=1.0, t_end=8.0)
    bar_q_externals = simproj.merge_events(meta, market_orders)

    ctx = simproj.ConditionalSimulationContext(
        process,
        [list(arr) for arr in cond_by_dim],
        10.0,
        cond_externals=market_orders,
        new_externals=bar_q_externals,
    )

    times = np.linspace(0.0, 10.0, 11).astype(np.float64)
    samples = ctx.simulate_queue_at_times(times, initial_queue_size=200, seed=0)
    assert samples.shape == (11,)
    assert samples.dtype == np.uint32
