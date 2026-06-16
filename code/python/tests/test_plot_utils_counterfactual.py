import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def _load_module(monkeypatch, tmp_path, name, relative_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_queue_fixture(path):
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "times.npy", np.array([0.0, 1.0]))
    np.save(path / "queue_paths.npy", np.array([[10, 12, 13], [11, 14, 15]], dtype=np.uint32))


def test_aggressive_plot_loader_labels_counterfactual_direction(monkeypatch, tmp_path):
    module = _load_module(
        monkeypatch,
        tmp_path,
        "plot_utils_propagator_test",
        "experiments/agressive_impact/load_experiments/plot_utils_propagator.py",
    )
    _write_queue_fixture(tmp_path)
    np.save(tmp_path / "impact_paths.npy", np.array([[0.0, 0.0], [1.0, 2.0]]))
    np.save(tmp_path / "event_types.npy", np.array([0.0, 1.0]))

    _, queue_with, _, _ = module.load_data(counterfactual=False, data_base=str(tmp_path))
    _, queue_without, _, _ = module.load_data(counterfactual=True, data_base=str(tmp_path))

    assert list(queue_with.columns) == ["q", "bar_q_sim_0", "bar_q_sim_1"]
    assert list(queue_without.columns) == ["bar_q", "q_sim_0", "q_sim_1"]


def test_aggressive_plot_queue_diff_keeps_bar_q_minus_q(monkeypatch, tmp_path):
    module = _load_module(
        monkeypatch,
        tmp_path,
        "plot_utils_propagator_diff_test",
        "experiments/agressive_impact/load_experiments/plot_utils_propagator.py",
    )

    with_df = pd.DataFrame([[10, 12, 13]], columns=["q", "bar_q_sim_0", "bar_q_sim_1"])
    without_df = pd.DataFrame([[12, 10, 9]], columns=["bar_q", "q_sim_0", "q_sim_1"])

    assert module._queue_diffs(with_df, counterfactual=False).iloc[0].tolist() == [2, 3]
    assert module._queue_diffs(without_df, counterfactual=True).iloc[0].tolist() == [2, 3]


def test_queue_plot_loader_labels_counterfactual_direction(monkeypatch, tmp_path):
    module = _load_module(
        monkeypatch,
        tmp_path,
        "queue_plot_utils_test",
        "experiments/queue_simulation/load_experiments/plot_utils.py",
    )
    _write_queue_fixture(tmp_path)

    queue_with = module.load_data(counterfactual=False, data_base=str(tmp_path))
    queue_without = module.load_data(counterfactual=True, data_base=str(tmp_path))

    assert list(queue_with.columns) == ["q", "bar_q_sim_0", "bar_q_sim_1"]
    assert list(queue_without.columns) == ["bar_q", "q_sim_0", "q_sim_1"]


def test_queue_plot_loader_prefers_scenario_subdirectories(monkeypatch, tmp_path):
    module = _load_module(
        monkeypatch,
        tmp_path,
        "queue_plot_utils_scenario_dirs_test",
        "experiments/queue_simulation/load_experiments/plot_utils.py",
    )
    monkeypatch.setattr(module, "SCRIPT_DIR", str(tmp_path))
    with_dir = tmp_path / "data" / "single" / "efficient" / "with"
    without_dir = tmp_path / "data" / "single" / "efficient" / "without"
    _write_queue_fixture(with_dir)
    without_dir.mkdir(parents=True, exist_ok=True)
    np.save(without_dir / "times.npy", np.array([0.0, 1.0]))
    np.save(
        without_dir / "queue_paths.npy",
        np.array([[20, 18, 17], [21, 19, 18]], dtype=np.uint32),
    )

    queue_with = module.load_data(counterfactual=False)
    queue_without = module.load_data(counterfactual=True)

    assert queue_with.iloc[0, 0] == 10
    assert queue_without.iloc[0, 0] == 20


def test_queue_plot_can_suppress_title(monkeypatch, tmp_path):
    module = _load_module(
        monkeypatch,
        tmp_path,
        "queue_plot_utils_no_title_test",
        "experiments/queue_simulation/load_experiments/plot_utils.py",
    )
    df = pd.DataFrame(
        [[10, 12], [11, 13]],
        index=pd.Index([0.0, 1.0], name="time"),
        columns=["q", "bar_q_sim_0"],
    )

    monkeypatch.setattr(module.plt, "show", lambda: None)
    module.plot_queue_shades(df, save_path=None, include_title=False)

    ax = module.plt.gcf().axes[0]
    assert ax.get_title() == ""
    module.plt.close("all")
