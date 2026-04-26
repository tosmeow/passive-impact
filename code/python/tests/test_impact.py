import numpy as np
import simproj


def test_tail_impact_from_affine():
    ti = simproj.TailImpact.from_affine_queue(
        mu=1.0, alpha=[0.065, 0.2], beta=[0.15, 0.6],
        b_l=-0.275, b_c=0.125, events=[1.0, 2.0, 3.0],
    )
    assert ti is not None  # opaque smoke


def test_aggressive_impact_from_samples():
    hawkes = simproj.MultiExponentialHawkes(
        mu=1.0, alpha=[0.065, 0.2], beta=[0.15, 0.6],
    )
    n = 10
    q = np.full(n, 200, dtype=np.uint32)
    q_bar = np.full(n, 180, dtype=np.uint32)
    times = np.linspace(0.0, 10.0, n).astype(np.float64)
    is_market = [True] * n

    def kappa(q):
        return 1000.0 * np.sqrt(np.log(np.exp(-0.01 * q) + 1.0))

    result = simproj.aggressive_impact_from_queue_samples(
        q_samples=q, q_bar_samples=q_bar,
        eval_times=times, is_market_order=is_market,
        hawkes=hawkes, kappa=kappa,
    )
    impact = result.impact()
    assert impact.shape == (n,)
    assert impact.dtype == np.float64
