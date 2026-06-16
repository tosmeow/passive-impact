"""Shared helpers for experiment plotting scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def script_dir(file: str) -> Path:
    """Return the directory containing a plotting script."""
    return Path(file).resolve().parent


def data_dir(file: str, *parts: str) -> Path:
    """Return a path under the script-local data directory."""
    return script_dir(file).joinpath("data", *parts)


def image_dir(file: str, dirname: str = "images") -> Path:
    """Return a path under the script-local image directory."""
    return script_dir(file) / dirname


def add_title_argument(parser, *, default: bool = False) -> None:
    """Add the standard title/no-title flag pair to an argparse parser."""
    title_group = parser.add_mutually_exclusive_group()
    title_group.add_argument(
        "--title",
        dest="include_title",
        action="store_true",
        help="Draw titles on generated PNG images.",
    )
    title_group.add_argument(
        "--no-title",
        dest="include_title",
        action="store_false",
        help="Do not draw titles on generated PNG images.",
    )
    parser.set_defaults(include_title=default)


def maybe_set_title(ax, title: str, include_title: bool) -> None:
    """Set an axes title only when titles are enabled."""
    if include_title:
        ax.set_title(title)


def save_or_show(fig, save_path=None, *, dpi: int = 300, bbox_inches: str = "tight") -> None:
    """Save a figure to disk or display it when no path is provided."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        print(f"Saved: {path}")
    else:
        plt.show()
