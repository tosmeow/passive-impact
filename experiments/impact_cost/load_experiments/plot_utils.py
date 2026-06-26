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
    from experiments.plot_utils_common import (
        add_format_argument,
        add_title_argument,
        with_output_format,
    )
    from experiments.impact_cost.load_experiments.lifecycle_passive_cost import (
        DEFAULT_CONFIG_PATH,
        _plot_lifecycle_paths,
        load_lifecycle_config,
    )
else:
    from ...plot_utils_common import (
        add_format_argument,
        add_title_argument,
        with_output_format,
    )
    from .lifecycle_passive_cost import (
        DEFAULT_CONFIG_PATH,
        _plot_lifecycle_paths,
        load_lifecycle_config,
    )


def generate_lifecycle_plot(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    include_title: bool = False,
    output_format: str = "png",
) -> Path:
    """Regenerate the canonical lifecycle impact-cost plot from saved summaries."""
    cfg = load_lifecycle_config(config_path)
    data_dir = Path(cfg.output_dir)
    image_dir = Path(cfg.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    cost_summary = _read_csv(data_dir / "impact_cost_path_summary.csv")
    impact_summary = _read_csv(data_dir / "price_impact_path_summary.csv")
    active_summary = _read_csv(data_dir / "active_quantity_path_summary.csv")

    plot_path = with_output_format(image_dir / "lifecycle_impact_cost_paths.png", output_format)

    _plot_lifecycle_paths(
        plot_path,
        pd.DataFrame(),
        cost_summary,
        impact_summary,
        active_summary,
        cfg,
        include_title=include_title,
    )
    return plot_path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run lifecycle_passive_cost.py before plotting."
        )
    return pd.read_csv(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    add_title_argument(parser, default=False)
    add_format_argument(parser, default="png")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        generate_lifecycle_plot(
            args.config,
            include_title=args.include_title,
            output_format=args.output_format,
        )
    )


if __name__ == "__main__":
    main()
