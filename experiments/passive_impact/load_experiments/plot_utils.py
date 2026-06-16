"""Plot utilities for passive impact experiments (single + double queue).

Loads pre-saved baseline data from ./data/{single,double}/{efficient,general}/{with,without}/.

Usage:
    python plot_utils.py --mode single --data-mode efficient --meta-end 60.0
    python plot_utils.py --mode double --data-mode efficient
    python plot_utils.py --mode single --title
"""
import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from plot_utils_single import (
        generate_all_plots as gen_single,
        load_data,
        plot_queue_shades,
    )
    from plot_utils_double import (
        generate_all_plots as gen_double,
        load_bidask_data,
        plot_dashboard,
        plot_impact_difference,
    )
    from experiments.plot_utils_common import add_format_argument, add_title_argument
else:
    from .plot_utils_single import (
        generate_all_plots as gen_single,
        load_data,
        plot_queue_shades,
    )
    from .plot_utils_double import (
        generate_all_plots as gen_double,
        load_bidask_data,
        plot_dashboard,
        plot_impact_difference,
    )
    from ...plot_utils_common import add_format_argument, add_title_argument


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate passive impact plots (single or double queue).'
    )
    p.add_argument('--mode', choices=['single', 'double'], default='single',
                   help='Queue model: single (one-sided) or double (bid-ask).')
    p.add_argument('--data-mode', choices=['general', 'efficient'], default='efficient',
                   help='Which simulation backend results to load (default: efficient).')
    p.add_argument('--meta-end', type=float, default=60.0,
                   help='Time at which the metaorder ends, drawn as a vertical line (default: 60.0). '
                        'Only used for single-queue mode.')
    add_title_argument(p, default=False)
    add_format_argument(p, default='pdf')
    return p.parse_args()


def generate_all_plots(mode, data_mode, meta_end=60.0, include_title=False, output_format='pdf'):
    """Generate all plots for the given mode.

    Args:
        mode: 'single' or 'double'
        data_mode: 'efficient' or 'general'
        meta_end: metaorder end time (only used in single-queue mode)
        include_title: whether to draw plot titles
    """
    if mode == 'single':
        gen_single(data_mode, meta_end, include_title=include_title, output_format=output_format)
    else:
        gen_double(data_mode, meta_end, include_title=include_title, output_format=output_format)


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(
        args.mode,
        args.data_mode,
        args.meta_end,
        include_title=args.include_title,
        output_format=args.output_format,
    )
