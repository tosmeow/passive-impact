"""Retarget config.toml propagator weights to a requested implied norm.

The base config supplies the propagator shape.  If

    xi(t) = 1 + sum_i a_i exp(-beta_i t),     a_i = weight_i / kappa,

then the implied Hawkes norm is n = A / (1 + A), where A = sum_i a_i.  This
script rescales the base a_i so A becomes n_star / (1 - n_star), then writes
the corresponding propagator_weights line in the target config.
"""
from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = HERE / "base_config.toml"
DEFAULT_TARGET_CONFIG = HERE / "config.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _impact_values(config: dict[str, Any], *, path: Path) -> tuple[float, list[float]]:
    try:
        impact = config["impact"]
        kappa = float(impact["propagator_kappa"])
        weights = [float(weight) for weight in impact["propagator_weights"]]
    except KeyError as exc:
        raise ValueError(f"{path} is missing impact.{exc.args[0]}") from exc

    if kappa == 0.0:
        raise ValueError(f"{path} has propagator_kappa = 0, so weights cannot be normalized")
    if not weights:
        raise ValueError(f"{path} has no propagator_weights")

    return kappa, weights


def transformed_weights(
    *,
    base_kappa: float,
    base_weights: list[float],
    target_kappa: float,
    implied_norm: float,
) -> tuple[list[float], float, float, float]:
    """Return target weights, base mass, target mass, and applied scale."""
    if not 0.0 <= implied_norm < 1.0:
        raise ValueError("implied_norm must satisfy 0 <= implied_norm < 1")

    base_xi_weights = [weight / base_kappa for weight in base_weights]
    base_mass = sum(base_xi_weights)
    if base_mass == 0.0:
        raise ValueError("base propagator has xi(0) - 1 = 0, so its shape cannot be rescaled")

    target_mass = implied_norm / (1.0 - implied_norm)
    scale = target_mass / base_mass
    new_weights = [target_kappa * weight * scale for weight in base_xi_weights]
    return new_weights, base_mass, target_mass, scale


def _format_weights(weights: list[float], *, digits: int) -> str:
    return "[" + ", ".join(f"{weight:.{digits}f}" for weight in weights) + "]"


def replace_weights_line(config_path: Path, rendered_weights: str, *, dry_run: bool) -> bool:
    text = config_path.read_text()
    pattern = re.compile(r"^(\s*propagator_weights\s*=\s*)\[[^\n]*\](\s*(?:#.*)?\n?)$", re.M)
    new_text, replacements = pattern.subn(rf"\1{rendered_weights}\2", text, count=1)
    if replacements != 1:
        raise ValueError(f"expected exactly one propagator_weights line in {config_path}")

    if not dry_run and text != new_text:
        config_path.write_text(new_text)
    return text != new_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "implied_norm",
        type=float,
        help="Target implied Hawkes norm n_*; must satisfy 0 <= n_* < 1.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help=f"Base TOML config used for the propagator shape. Default: {DEFAULT_BASE_CONFIG}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TARGET_CONFIG,
        help=f"Target TOML config whose propagator_weights line is rewritten. Default: {DEFAULT_TARGET_CONFIG}",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=8,
        help="Number of fixed decimal places to write for each weight. Default: 8.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the transformed weights without editing config.toml.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.digits < 0:
        raise SystemExit("--digits must be non-negative")

    base_config = _load_toml(args.base_config)
    target_config = _load_toml(args.config)
    base_kappa, base_weights = _impact_values(base_config, path=args.base_config)
    target_kappa, _ = _impact_values(target_config, path=args.config)

    weights, base_mass, target_mass, scale = transformed_weights(
        base_kappa=base_kappa,
        base_weights=base_weights,
        target_kappa=target_kappa,
        implied_norm=args.implied_norm,
    )
    rendered_weights = _format_weights(weights, digits=args.digits)
    changed = replace_weights_line(args.config, rendered_weights, dry_run=args.dry_run)

    action = "Would update" if args.dry_run else "Updated"
    if not changed and not args.dry_run:
        action = "Left unchanged"
    print(f"{action}: {args.config}")
    print(f"base_config: {args.base_config}")
    print(f"target_kappa: {target_kappa:.12g}")
    print(f"base_xi_mass: {base_mass:.12g}")
    print(f"target_xi_mass: {target_mass:.12g}")
    print(f"scale: {scale:.12g}")
    print(f"target_xi0: {1.0 + target_mass:.12g}")
    print(f"target_implied_norm: {args.implied_norm:.12g}")
    print(f"propagator_weights = {rendered_weights}")


if __name__ == "__main__":
    main()
