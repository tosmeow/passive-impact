import numpy as np
from simproj.queue_simulation import QueueSimulationConfig, run, save


def test_queue_simulation_smoke(tmp_path):
    cfg = QueueSimulationConfig(
        time_horizon=2.0, n_simulations=2, n_eval_times=20,
        initial_queue_size=200, mode="single",
        metaorder=4, metaorder_window=(0.1, 1.5), seed=42,
    )
    result = run(cfg)
    assert result["queue_paths"].shape == (20, 3)
    save(result, str(tmp_path))
    assert (tmp_path / "queue_paths.npy").exists()


def test_queue_simulation_counterfactual_flag_and_save(tmp_path):
    base_kwargs = dict(
        time_horizon=5.0,
        n_simulations=2,
        n_eval_times=50,
        initial_queue_size=200,
        mode="single",
        metaorder=20,
        metaorder_window=(0.1, 4.0),
        seed=42,
    )

    with_result = run(QueueSimulationConfig(**base_kwargs, counterfactual=False))
    without_result = run(QueueSimulationConfig(**base_kwargs, counterfactual=True))

    assert with_result["queue_paths"].shape == (50, 3)
    assert without_result["queue_paths"].shape == (50, 3)
    assert not np.array_equal(
        with_result["queue_paths"][:, 0],
        without_result["queue_paths"][:, 0],
    )

    save(with_result, str(tmp_path))
    assert (tmp_path / "queue_paths.npy").exists()
