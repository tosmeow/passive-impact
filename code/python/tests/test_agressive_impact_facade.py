import numpy as np
from simproj.agressive_impact import AggressiveImpactConfig, _make_meta_orders, run, save


def test_agressive_impact_smoke(tmp_path):
    cfg = AggressiveImpactConfig(
        time_horizon=2.0, n_simulations=2, initial_queue_size=200,
        metaorder=4, metaorder_window=(0.1, 1.5),
        seed=42,
    )
    result = run(cfg)
    assert result["queue_paths"].shape[1] == 3
    assert result["impact_paths"].shape[1] == 2
    assert "event_types" in result
    assert "bar_kappa" in result

    save(result, str(tmp_path))
    assert (tmp_path / "queue_paths.npy").exists()
    assert (tmp_path / "event_types.npy").exists()
    assert (tmp_path / "bar_kappa.npy").exists()


def test_agressive_explicit_metaorder_times_are_market_orders():
    cfg = AggressiveImpactConfig(metaorder=np.array([0.1, 0.4, 0.9], dtype=np.float64))
    meta = _make_meta_orders(cfg)
    assert np.all(meta.dims() == 2)


def test_agressive_impact_accepts_custom_bar_kappa(tmp_path):
    cfg = AggressiveImpactConfig(
        time_horizon=2.0, n_simulations=2, initial_queue_size=200,
        bar_kappa=10.0,
        metaorder=4, metaorder_window=(0.1, 1.5),
        seed=42,
    )
    result = run(cfg)
    assert result["impact_paths"].shape[1] == 2
    assert result["bar_kappa"][0] == 10.0


def test_agressive_impact_counterfactual_flag_and_save(tmp_path):
    base_kwargs = dict(
        time_horizon=5.0,
        n_simulations=2,
        initial_queue_size=200,
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
