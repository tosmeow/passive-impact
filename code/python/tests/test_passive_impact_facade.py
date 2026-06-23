import numpy as np
from simproj.passive_impact import PassiveImpactConfig, _make_meta_orders, run, save


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


def test_passive_explicit_metaorder_times_are_limit_orders():
    cfg = PassiveImpactConfig(metaorder=np.array([0.1, 0.4, 0.9], dtype=np.float64))
    meta = _make_meta_orders(cfg)
    assert np.all(meta.dims() == 0)


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


def test_passive_impact_double_smoke():
    cfg = PassiveImpactConfig(
        time_horizon=5.0,
        n_simulations=2,
        initial_queue_size=200,
        mode="double",
        metaorder=20,
        metaorder_window=(0.1, 4.0),
        b_l_cross=0.05,
        b_c_cross=0.02,
        seed=42,
    )
    result = run(cfg)

    assert result["ask_queue_paths"].shape[1] == 3
    assert result["bid_queue_paths"].shape[1] == 3
    assert result["ask_impact_paths"].shape[1] == 2
    assert result["bid_impact_paths"].shape[1] == 2
    assert result["ask_queue_paths"].shape[0] == len(result["ask_times"])
    assert result["bid_queue_paths"].shape[0] == len(result["bid_times"])


def test_passive_impact_double_with_without_use_opposite_conditioning_paths():
    base_kwargs = dict(
        time_horizon=5.0,
        n_simulations=2,
        initial_queue_size=200,
        mode="double",
        metaorder=20,
        metaorder_window=(0.1, 4.0),
        b_l_cross=0.05,
        b_c_cross=0.02,
        seed=42,
    )

    with_result = run(PassiveImpactConfig(**base_kwargs, counterfactual=False))
    without_result = run(PassiveImpactConfig(**base_kwargs, counterfactual=True))

    assert with_result["ask_queue_paths"].shape == without_result["ask_queue_paths"].shape
    assert with_result["bid_queue_paths"].shape == without_result["bid_queue_paths"].shape
    assert not np.array_equal(
        with_result["ask_queue_paths"][:, 0],
        without_result["ask_queue_paths"][:, 0],
    )


def test_passive_impact_scales_by_c_kappa_effective():
    base_kwargs = dict(
        time_horizon=3.0,
        n_simulations=2,
        initial_queue_size=200,
        mode="single",
        counterfactual=False,
        metaorder=8,
        metaorder_window=(0.1, 2.5),
        seed=42,
    )

    normalized = run(PassiveImpactConfig(**base_kwargs, c_kappa_effective=1.0))
    scaled = run(PassiveImpactConfig(**base_kwargs, c_kappa_effective=-0.25))

    np.testing.assert_array_equal(normalized["times"], scaled["times"])
    np.testing.assert_array_equal(normalized["queue_paths"], scaled["queue_paths"])
    np.testing.assert_allclose(
        scaled["impact_paths"],
        -0.25 * normalized["impact_paths"],
    )


def test_passive_impact_bid_side_flips_impact_sign():
    base_kwargs = dict(
        time_horizon=3.0,
        n_simulations=2,
        initial_queue_size=200,
        mode="single",
        counterfactual=False,
        metaorder=8,
        metaorder_window=(0.1, 2.5),
        seed=42,
    )

    ask = run(PassiveImpactConfig(**base_kwargs, side="ask"))
    bid = run(PassiveImpactConfig(**base_kwargs, side="bid"))

    np.testing.assert_array_equal(ask["times"], bid["times"])
    np.testing.assert_array_equal(ask["queue_paths"], bid["queue_paths"])
    np.testing.assert_allclose(bid["impact_paths"], -ask["impact_paths"])


def test_passive_impact_rejects_nonfinite_c_kappa_effective():
    cfg = PassiveImpactConfig(time_horizon=1.0, n_simulations=1, c_kappa_effective=np.nan)

    try:
        run(cfg)
    except ValueError as exc:
        assert "c_kappa_effective" in str(exc)
    else:
        raise AssertionError("expected nonfinite c_kappa_effective to be rejected")


def test_passive_impact_rejects_unknown_side():
    cfg = PassiveImpactConfig(time_horizon=1.0, n_simulations=1, side="middle")

    try:
        run(cfg)
    except ValueError as exc:
        assert "side" in str(exc)
    else:
        raise AssertionError("expected unknown side to be rejected")
