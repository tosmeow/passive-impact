import numpy as np
from simproj.passive_impact import PassiveImpactConfig, run, save


def test_passive_impact_smoke(tmp_path):
    cfg = PassiveImpactConfig(
        time_horizon=2.0,
        n_simulations=2,
        initial_queue_size=200,
        mode="single",
        counterfactual=False,
        metaorder=4,
        metaorder_window=(0.1, 1.5),
        seed=42,
    )
    result = run(cfg)
    assert "times" in result
    assert "queue_paths" in result
    assert "impact_paths" in result
    assert result["queue_paths"].shape[1] == 3  # q + 2 sims
    assert result["impact_paths"].shape[1] == 2

    save(result, str(tmp_path))
    assert (tmp_path / "times.npy").exists()
    assert (tmp_path / "queue_paths.npy").exists()
    assert (tmp_path / "impact_paths.npy").exists()

    # Impact paths should now be non-trivial (not all zeros)
    assert np.any(result["impact_paths"] != 0.0)


def test_passive_impact_with_without_use_opposite_conditioning_paths():
    base_kwargs = dict(
        time_horizon=5.0,
        n_simulations=2,
        initial_queue_size=200,
        mode="single",
        metaorder=20,
        metaorder_window=(0.1, 4.0),
        seed=42,
    )

    with_result = run(PassiveImpactConfig(**base_kwargs, counterfactual=False))
    without_result = run(PassiveImpactConfig(**base_kwargs, counterfactual=True))

    assert with_result["queue_paths"].shape == without_result["queue_paths"].shape
    assert with_result["impact_paths"].shape == without_result["impact_paths"].shape

    # counterfactual=False: first column is q; True: first column is bar_q.
    assert not np.array_equal(
        with_result["queue_paths"][:, 0],
        without_result["queue_paths"][:, 0],
    )
