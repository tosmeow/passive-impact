"""Shared helpers for experiment plotting scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_OUTPUT_FORMAT = "pdf"
OUTPUT_FORMATS = ("pdf", "png")


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
        help="Draw titles on generated plot files.",
    )
    title_group.add_argument(
        "--no-title",
        dest="include_title",
        action="store_false",
        help="Do not draw titles on generated plot files.",
    )
    parser.set_defaults(include_title=default)


def add_format_argument(parser, *, default: str = DEFAULT_OUTPUT_FORMAT) -> None:
    """Add the standard output-format selector to an argparse parser."""
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=OUTPUT_FORMATS,
        default=default,
        help=f"Output file format (default: {default}).",
    )


def normalize_output_format(output_format: str | None) -> str:
    """Normalize and validate an output format string."""
    fmt = (output_format or DEFAULT_OUTPUT_FORMAT).lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}; got {output_format!r}")
    return fmt


def with_output_format(path, output_format: str | None) -> Path:
    """Return `path` with its suffix replaced by the requested output format."""
    return Path(path).with_suffix(f".{normalize_output_format(output_format)}")


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
