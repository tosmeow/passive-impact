import numpy as np
import simproj


def test_hawkes_stationary_state():
    h = simproj.MultiExponentialHawkes.with_stationary_state(
        mu=1.0, alpha=[0.065, 0.2], beta=[0.15, 0.6],
    )
    state = h.stationary_state()
    assert state.shape == (2,)
    assert np.all(state > 0)


def test_hawkes_simulate_returns_event_times():
    h = simproj.MultiExponentialHawkes(
        mu=1.0, alpha=[0.065, 0.2, 0.325, 0.65], beta=[0.15, 0.6, 2.5, 10.0],
    )
    times = simproj.simulate_hawkes(h, t_max=10.0, seed=42)
    assert times.dtype == np.float64
    assert times.ndim == 1
    assert len(times) > 0
    assert np.all(np.diff(times) >= 0)  # strictly increasing
