"""Regenerate canonical lifecycle impact-cost figures from saved CSV outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from experiments.impact_cost.load_experiments.lifecycle_passive_cost import (
        DEFAULT_CONFIG_PATH,
        _plot_lifecycle_paths,
        _plot_representative_step_paths,
        load_lifecycle_config,
    )
else:
    from .lifecycle_passive_cost import (
        DEFAULT_CONFIG_PATH,
        _plot_lifecycle_paths,
        _plot_representative_step_paths,
        load_lifecycle_config,
    )


def generate_all_plots(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    include_title: bool = True,
) -> list[Path]:
    """Regenerate all canonical lifecycle plots from saved output tables."""
    cfg = load_lifecycle_config(config_path)
    data_dir = Path(cfg.output_dir)
    image_dir = Path(cfg.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_csv(data_dir / "impact_cost_path_samples.csv")
    cost_summary = _read_csv(data_dir / "impact_cost_path_summary.csv")
    impact_summary = _read_csv(data_dir / "price_impact_path_summary.csv")
    active_summary = _read_csv(data_dir / "active_quantity_path_summary.csv")
    jumps = _read_csv(data_dir / "impact_cost_fill_jumps.csv")
    path_summary = _read_csv(data_dir / "policy_path_summary.csv")

    plot_path = image_dir / "lifecycle_impact_cost_paths.png"
    step_path = image_dir / "lifecycle_representative_cost_steps.png"
    shared_step_path = image_dir / "lifecycle_representative_cost_steps_shared_y.png"

    _plot_lifecycle_paths(
        plot_path,
        samples,
        cost_summary,
        impact_summary,
        active_summary,
        cfg,
        include_title=include_title,
    )
    _plot_representative_step_paths(
        step_path,
        samples,
        jumps,
        path_summary,
        cfg,
        include_title=include_title,
    )
    _plot_representative_step_paths(
        shared_step_path,
        samples,
        jumps,
        path_summary,
        cfg,
        shared_y=True,
        include_title=include_title,
    )
    return [plot_path, step_path, shared_step_path]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run lifecycle_passive_cost.py before plotting."
        )
    return pd.read_csv(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw titles on generated PNG images.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in generate_all_plots(args.config, include_title=not args.no_title):
        print(path)


if __name__ == "__main__":
    main()
