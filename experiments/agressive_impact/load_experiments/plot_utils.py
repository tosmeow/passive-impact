"""Plot utilities for aggressive impact experiments (propagator + hybrid).

Loads pre-saved baseline data from:
    ./data/propagator/   — pure-propagator model
    ./data/hybrid/       — hybrid propagator+instantaneous model

Usage:
    python plot_utils.py --model propagator
    python plot_utils.py --model hybrid
    python plot_utils.py --model propagator --counterfactual
"""
import argparse

from plot_utils_propagator import (
    load_data as _load_propagator_data,
    plot_impact_by_event_type,
    plot_queue_diff,
    plot_shades,
)
from plot_utils_propagator import generate_all_plots as gen_propagator
from plot_utils_hybrid import generate_all_plots as gen_hybrid


def load_data(counterfactual=False):
    """Load propagator data using the legacy 3-value notebook API."""
    impact_df, queue_df, is_market, _meta_end = _load_propagator_data(
        counterfactual=counterfactual
    )
    return impact_df, queue_df, is_market


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate aggressive impact plots (propagator or hybrid model).'
    )
    p.add_argument('--model', choices=['propagator', 'hybrid'], default='propagator',
                   help='Which aggressive impact model results to plot.')
    p.add_argument('--counterfactual', action='store_true',
                   help='Interpret queue_paths.npy as without-us output: first column bar_q, simulations q.')
    p.add_argument('--data-base', default=None,
                   help='Directory containing times.npy, queue_paths.npy, impact_paths.npy, and event_types.npy.')
    p.add_argument('--output-dir', default=None,
                   help='Directory where images should be written.')
    p.add_argument('--bar-kappa', type=float, default=None,
                   help='Hybrid-only value used when data-base does not contain bar_kappa.npy.')
    return p.parse_args()


def generate_all_plots(model, counterfactual=False, data_base=None, output_dir=None, bar_kappa=None):
    """Generate all plots for the given model.

    Args:
        model: 'propagator' or 'hybrid'
    """
    if model == 'propagator':
        gen_propagator(
            counterfactual=counterfactual,
            data_base=data_base,
            output_dir=output_dir,
        )
    else:
        gen_hybrid(
            counterfactual=counterfactual,
            data_base=data_base,
            output_dir=output_dir,
            bar_kappa=bar_kappa,
        )


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(
        args.model,
        counterfactual=args.counterfactual,
        data_base=args.data_base,
        output_dir=args.output_dir,
        bar_kappa=args.bar_kappa,
    )
