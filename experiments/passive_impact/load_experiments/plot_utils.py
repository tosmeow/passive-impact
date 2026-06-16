"""Plot utilities for passive impact experiments (single + double queue).

Loads pre-saved baseline data from ./data/{single,double}/{efficient,general}/{with,without}/.

Usage:
    python plot_utils.py --mode single --data-mode efficient --meta-end 80.0
    python plot_utils.py --mode double --data-mode efficient
    python plot_utils.py --mode single --no-title
"""
import argparse

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


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate passive impact plots (single or double queue).'
    )
    p.add_argument('--mode', choices=['single', 'double'], default='single',
                   help='Queue model: single (one-sided) or double (bid-ask).')
    p.add_argument('--data-mode', choices=['general', 'efficient'], default='efficient',
                   help='Which simulation backend results to load (default: efficient).')
    p.add_argument('--meta-end', type=float, default=80.0,
                   help='Time at which the metaorder ends, drawn as a vertical line (default: 80.0). '
                        'Only used for single-queue mode.')
    p.add_argument('--no-title', action='store_true',
                   help='Do not draw titles on generated PNG images.')
    return p.parse_args()


def generate_all_plots(mode, data_mode, meta_end=80.0, include_title=True):
    """Generate all plots for the given mode.

    Args:
        mode: 'single' or 'double'
        data_mode: 'efficient' or 'general'
        meta_end: metaorder end time (only used in single-queue mode)
        include_title: whether to draw plot titles
    """
    if mode == 'single':
        gen_single(data_mode, meta_end, include_title=include_title)
    else:
        gen_double(data_mode, meta_end, include_title=include_title)


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(
        args.mode,
        args.data_mode,
        args.meta_end,
        include_title=not args.no_title,
    )
