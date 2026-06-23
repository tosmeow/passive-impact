"""Plot utilities for conditional point-process perturbation outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiments.plot_utils_common import (
    add_format_argument,
    add_title_argument,
    maybe_set_title,
    save_or_show,
    script_dir,
    with_output_format,
)

SCRIPT_DIR = script_dir(__file__)
PROCESS_KINDS = ("affine", "hawkes")
REQUIRED_OUTPUTS = [
    "baseline_times.npy",
    "perturbation_times.npy",
    "perturbed_times.npy",
    "perturbed_lengths.npy",
]


def _default_data_dir() -> Path:
    return Path(SCRIPT_DIR) / "output"


def _default_output_dir() -> Path:
    return Path(SCRIPT_DIR) / "images"


def _has_point_process_outputs(base: Path) -> bool:
    return all((base / name).exists() for name in REQUIRED_OUTPUTS)


def _available_data_dirs(data_base=None) -> list[Path]:
    base = Path(data_base) if data_base is not None else _default_data_dir()
    data_dirs = [base / kind for kind in PROCESS_KINDS if _has_point_process_outputs(base / kind)]
    if data_dirs:
        return data_dirs

    if _has_point_process_outputs(base):
        return [base]

    raise FileNotFoundError(f"No point-process outputs found under {base}")


def load_data(data_base=None) -> dict:
    """Load point-process simulation outputs from .npy files."""
    base = Path(data_base) if data_base is not None else _default_data_dir()
    missing = [name for name in REQUIRED_OUTPUTS if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under {base}")

    data = {
        "baseline_times": np.load(base / "baseline_times.npy"),
        "perturbation_times": np.load(base / "perturbation_times.npy"),
        "perturbed_times": np.load(base / "perturbed_times.npy"),
        "perturbed_lengths": np.load(base / "perturbed_lengths.npy"),
    }
    horizon_path = base / "time_horizon.npy"
    if horizon_path.exists():
        data["time_horizon"] = float(np.load(horizon_path)[0])
    kind_path = base / "process_kind.npy"
    if kind_path.exists():
        data["process_kind"] = str(np.load(kind_path)[0])
    return data


def _process_label(data: dict, data_dir: Path) -> str:
    kind = data.get("process_kind")
    if kind:
        return str(kind)
    if data_dir.name in PROCESS_KINDS:
        return data_dir.name
    return "point_process"


def _path_rows(perturbed_times: np.ndarray, perturbed_lengths: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(perturbed_times[row, : int(length)], dtype=np.float64)
        for row, length in enumerate(perturbed_lengths)
    ]


def _infer_horizon(data: dict, time_horizon=None) -> float:
    if time_horizon is not None:
        return float(time_horizon)
    if "time_horizon" in data:
        return float(data["time_horizon"])

    maxima = []
    for arr in (data["baseline_times"], data["perturbation_times"]):
        if len(arr):
            maxima.append(float(np.nanmax(arr)))
    for path in _path_rows(data["perturbed_times"], data["perturbed_lengths"]):
        if len(path):
            maxima.append(float(np.nanmax(path)))
    return max(maxima, default=1.0)


def _step_xy(event_times: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(event_times, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    clean = clean[(0.0 <= clean) & (clean <= horizon)]
    x = np.concatenate(([0.0], clean, [horizon]))
    y = np.arange(len(x), dtype=float)
    y[-1] = y[-2] if len(y) > 1 else 0.0
    return x, y


def _path_colors(n_paths: int, alpha: float):
    cmap = plt.get_cmap("Blues")
    stops = np.linspace(0.35, 0.90, max(n_paths, 1))
    return [(*cmap(stop)[:3], alpha) for stop in stops[:n_paths]]


def plot_counting_process_shades(
    data: dict,
    *,
    time_horizon=None,
    path_alpha: float = 0.12,
    save_path=None,
    include_title: bool = False,
):
    """Plot the factual counting process and all shocked conditional paths."""
    if not (0.0 < path_alpha <= 1.0):
        raise ValueError("path_alpha must lie in (0, 1]")

    horizon = _infer_horizon(data, time_horizon=time_horizon)
    paths = _path_rows(data["perturbed_times"], data["perturbed_lengths"])
    if not paths:
        raise ValueError("No shocked paths available to plot")

    fig, ax = plt.subplots(figsize=(12, 6))

    for path, color in zip(paths, _path_colors(len(paths), path_alpha)):
        x, y = _step_xy(path, horizon)
        ax.step(x, y, where="post", color=color, linewidth=0.7)

    ax.add_line(
        Line2D(
            [],
            [],
            color="tab:blue",
            alpha=0.60,
            linewidth=1.2,
            label="Shocked conditional paths",
        )
    )

    factual_x, factual_y = _step_xy(data["baseline_times"], horizon)
    ax.step(
        factual_x,
        factual_y,
        where="post",
        color="black",
        linewidth=2.4,
        label="Factual path",
    )

    for idx, t in enumerate(data["perturbation_times"]):
        label = "Perturbation event" if idx == 0 else None
        ax.axvline(
            float(t),
            color="tab:green",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=label,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Event count")
    ax.set_xlim(0.0, horizon)
    kind = data.get("process_kind", "point-process").replace("_", " ")
    maybe_set_title(ax, f"Conditional {kind} paths after event perturbation", include_title)
    ax.legend()
    plt.tight_layout()
    save_or_show(fig, save_path, dpi=300)


def generate_all_plots(
    *,
    data_base=None,
    output_dir=None,
    time_horizon=None,
    path_alpha: float = 0.12,
    include_title: bool = False,
    output_format: str = "png",
):
    data_dirs = _available_data_dirs(data_base=data_base)
    output_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
    for data_dir in data_dirs:
        data = load_data(data_base=data_dir)
        label = _process_label(data, data_dir)
        save_dir = output_dir / label if len(data_dirs) > 1 else output_dir
        plot_counting_process_shades(
            data,
            time_horizon=time_horizon,
            path_alpha=path_alpha,
            save_path=with_output_format(
                save_dir / f"point_process_shocked_paths_{label}.png", output_format,
            ),
            include_title=include_title,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate point-process perturbation plots.")
    parser.add_argument(
        "--data-base",
        default=None,
        help="Directory containing the point-process .npy outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where images should be written.",
    )
    parser.add_argument(
        "--time-horizon",
        type=float,
        default=None,
        help="Plot horizon; defaults to saved/inferred horizon.",
    )
    parser.add_argument(
        "--path-alpha",
        type=float,
        default=0.12,
        help="Alpha for each shocked path line.",
    )
    add_title_argument(parser, default=False)
    add_format_argument(parser, default="png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_plots(
        data_base=args.data_base,
        output_dir=args.output_dir,
        time_horizon=args.time_horizon,
        path_alpha=args.path_alpha,
        include_title=args.include_title,
        output_format=args.output_format,
    )
