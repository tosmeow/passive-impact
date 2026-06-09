"""Run the lifecycle passive-cost pipeline from a saved JSON config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .lifecycle_passive_cost_pipeline import (
    LifecyclePassiveCostConfig,
    run_lifecycle_passive_cost_pipeline,
)


def load_config(
    config_path: str | Path, *, output_dir: str | None = None
) -> LifecyclePassiveCostConfig:
    """Load a saved lifecycle config, optionally replacing its output directory."""
    raw: dict[str, Any] = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if output_dir is not None:
        raw["output_dir"] = output_dir

    for key in ("propagator_weights", "propagator_beta", "hawkes_alpha", "hawkes_beta"):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(raw[key])

    return LifecyclePassiveCostConfig(**raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Path to lifecycle_passive_cost_config.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory stored in the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config_path, output_dir=args.output_dir)
    summary = run_lifecycle_passive_cost_pipeline(cfg)
    print(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in summary.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
