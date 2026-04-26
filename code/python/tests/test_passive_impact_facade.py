import numpy as np
from simproj.passive_impact import PassiveImpactConfig, run, save


def test_passive_impact_smoke(tmp_path):
    cfg = PassiveImpactConfig(
        time_horizon=2.0,
        n_simulations=2,
        initial_queue_size=200,
        mode="single",
        side="with",
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
