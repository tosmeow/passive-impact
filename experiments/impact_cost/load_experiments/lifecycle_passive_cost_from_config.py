"""Run the lifecycle passive-cost experiment from a saved TOML or JSON config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lifecycle_passive_cost import (
    LifecyclePassiveCostConfig,
    load_lifecycle_config,
    run_lifecycle_passive_cost_pipeline,
)


def load_config(
    config_path: str | Path,
    *,
    output_dir: str | None = None,
    image_dir: str | None = None,
) -> LifecyclePassiveCostConfig:
    """Load a lifecycle config, optionally replacing its output directory."""
    return load_lifecycle_config(
        config_path,
        overrides={"output_dir": output_dir, "image_dir": image_dir},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Path to config.toml or a saved JSON config")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory stored in the config.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Override the image directory stored in the config.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw titles on generated PNG images.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(
        args.config_path,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
    )
    summary = run_lifecycle_passive_cost_pipeline(
        cfg,
        include_title=not args.no_title,
    )
    print(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in summary.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
