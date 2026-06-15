import numpy as np
from simproj.agressive_impact import AggressiveImpactConfig, run, save


def test_agressive_impact_propagator_smoke(tmp_path):
    cfg = AggressiveImpactConfig(
        time_horizon=2.0, n_simulations=2, initial_queue_size=200,
        model="propagator",
        metaorder=4, metaorder_window=(0.1, 1.5),
        seed=42,
    )
    result = run(cfg)
    assert result["queue_paths"].shape[1] == 3
    assert result["impact_paths"].shape[1] == 2
    assert "event_types" in result

    save(result, str(tmp_path))
    assert (tmp_path / "queue_paths.npy").exists()
    assert (tmp_path / "event_types.npy").exists()


def test_agressive_impact_hybrid_smoke(tmp_path):
    cfg = AggressiveImpactConfig(
        time_horizon=2.0, n_simulations=2, initial_queue_size=200,
        model="hybrid", bar_kappa=10.0,
        metaorder=4, metaorder_window=(0.1, 1.5),
        seed=42,
    )
    result = run(cfg)
    assert result["impact_paths"].shape[1] == 2


def test_agressive_impact_counterfactual_flag_and_save(tmp_path):
    base_kwargs = dict(
        time_horizon=5.0,
        n_simulations=2,
        initial_queue_size=200,
        model="propagator",
        metaorder=20,
        metaorder_window=(0.1, 4.0),
        seed=42,
    )

    with_result = run(AggressiveImpactConfig(**base_kwargs, counterfactual=False))
    without_result = run(AggressiveImpactConfig(**base_kwargs, counterfactual=True))

    assert with_result["queue_paths"].shape == without_result["queue_paths"].shape
    assert with_result["impact_paths"].shape == without_result["impact_paths"].shape
    assert not np.array_equal(
        with_result["queue_paths"][:, 0],
        without_result["queue_paths"][:, 0],
    )

    save(with_result, str(tmp_path))
    assert (tmp_path / "queue_paths.npy").exists()
