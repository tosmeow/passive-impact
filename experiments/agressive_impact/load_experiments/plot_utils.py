"""Plot utilities for aggressive impact experiments (propagator + hybrid).

Loads pre-saved baseline data from:
    ./data/propagator/   — pure-propagator model
    ./data/hybrid/       — hybrid propagator+instantaneous model

Usage:
    python plot_utils.py --model propagator
    python plot_utils.py --model hybrid
"""
import argparse

from plot_utils_propagator import generate_all_plots as gen_propagator
from plot_utils_hybrid import generate_all_plots as gen_hybrid


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate aggressive impact plots (propagator or hybrid model).'
    )
    p.add_argument('--model', choices=['propagator', 'hybrid'], default='propagator',
                   help='Which aggressive impact model results to plot.')
    return p.parse_args()


def generate_all_plots(model):
    """Generate all plots for the given model.

    Args:
        model: 'propagator' or 'hybrid'
    """
    if model == 'propagator':
        gen_propagator()
    else:
        gen_hybrid()


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(args.model)
