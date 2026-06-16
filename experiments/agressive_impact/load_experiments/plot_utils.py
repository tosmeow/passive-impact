"""Plot utilities for aggressive impact experiments.

Loads generated data from:
    ./data/with/
    ./data/without/

Usage:
    python plot_utils.py
    python plot_utils.py --scenario without
    python plot_utils.py --title
"""
import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.plot_utils_aggressive import (
        compute_plot_y_lims,
        generate_all_plots as _generate_scenario_plots,
        load_data as _load_data,
        plot_impact_decomposition,
        plot_queue_diff,
        plot_shades,
    )
    from experiments.plot_utils_common import add_format_argument, add_title_argument
else:
    from .utils.plot_utils_aggressive import (
        compute_plot_y_lims,
        generate_all_plots as _generate_scenario_plots,
        load_data as _load_data,
        plot_impact_decomposition,
        plot_queue_diff,
        plot_shades,
    )
    from ...plot_utils_common import add_format_argument, add_title_argument


def load_data(counterfactual=False, data_base=None, bar_kappa=None):
    """Load aggressive impact data using the legacy 3-value notebook API."""
    impact_df, queue_df, is_market, _meta_end, _bar_kappa = _load_data(
        counterfactual=counterfactual,
        data_base=data_base,
        bar_kappa=bar_kappa,
    )
    return impact_df, queue_df, is_market


def plot_impact_by_event_type(
    impact_df,
    is_market,
    bar_kappa=0.01,
    meta_end=None,
    save_path=None,
    include_title=False,
):
    """Backward-compatible wrapper around the event decomposition plot."""
    return plot_impact_decomposition(
        impact_df,
        is_market,
        bar_kappa=bar_kappa,
        meta_end=meta_end,
        save_path=save_path,
        include_title=include_title,
    )


def parse_args():
    p = argparse.ArgumentParser(description='Generate aggressive impact plots.')
    p.add_argument('--scenario', choices=['with', 'without', 'both'], default=None,
                   help='Which conditioning case to plot (default: both).')
    p.add_argument('--counterfactual', action='store_true',
                   help='Alias for --scenario without: first column bar_q, simulations q.')
    p.add_argument('--data-base', default=None,
                   help='Directory containing saved .npy files, or with/without subdirectories.')
    p.add_argument('--output-dir', default=None,
                   help='Directory where images should be written.')
    p.add_argument('--bar-kappa', type=float, default=None,
                   help='Value used when data-base does not contain bar_kappa.npy.')
    add_title_argument(p, default=False)
    add_format_argument(p, default='pdf')
    args = p.parse_args()
    if args.counterfactual and args.scenario not in {None, 'without'}:
        p.error('--counterfactual cannot be combined with --scenario with or --scenario both')
    return args


def generate_all_plots(
    counterfactual=False,
    scenario=None,
    data_base=None,
    output_dir=None,
    bar_kappa=None,
    include_title=False,
    output_format='pdf',
):
    """Generate all plots for the requested conditioning scenario(s)."""
    scenario_counterfactuals = _scenario_counterfactuals(scenario, counterfactual)
    data_bases = {
        scenario_counterfactual: _scenario_data_base(data_base, scenario_counterfactual)
        for scenario_counterfactual in scenario_counterfactuals
    }
    y_lims = _shared_y_lims(
        compute_plot_y_lims(
            counterfactual=scenario_counterfactual,
            data_base=data_bases[scenario_counterfactual],
            bar_kappa=bar_kappa,
        )
        for scenario_counterfactual in scenario_counterfactuals
    )
    for scenario_counterfactual in scenario_counterfactuals:
        _generate_scenario_plots(
            counterfactual=scenario_counterfactual,
            data_base=data_bases[scenario_counterfactual],
            output_dir=output_dir,
            bar_kappa=bar_kappa,
            include_title=include_title,
            y_lims=y_lims,
            output_format=output_format,
        )


def _scenario_counterfactuals(scenario, counterfactual):
    if scenario is None:
        return [True] if counterfactual else [False, True]
    if scenario == 'with':
        return [False]
    if scenario == 'without':
        return [True]
    if scenario == 'both':
        return [False, True]
    raise ValueError("scenario must be one of None, 'with', 'without', or 'both'")


def _scenario_data_base(data_base, counterfactual):
    if data_base is None:
        return None
    path = Path(data_base)
    if _has_plot_arrays(path):
        return path
    scenario_path = path / ('without' if counterfactual else 'with')
    if _has_plot_arrays(scenario_path):
        return scenario_path
    return path


def _has_plot_arrays(path):
    return (
        (path / 'times.npy').exists()
        and (path / 'queue_paths.npy').exists()
        and (path / 'impact_paths.npy').exists()
        and (path / 'event_types.npy').exists()
    )


def _shared_y_lims(y_lims_by_scenario):
    shared = {}
    for y_lims in y_lims_by_scenario:
        for name, (ymin, ymax) in y_lims.items():
            if name in shared:
                old_ymin, old_ymax = shared[name]
                shared[name] = min(old_ymin, ymin), max(old_ymax, ymax)
            else:
                shared[name] = ymin, ymax
    return shared


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(
        counterfactual=args.counterfactual,
        scenario=args.scenario,
        data_base=args.data_base,
        output_dir=args.output_dir,
        bar_kappa=args.bar_kappa,
        include_title=args.include_title,
        output_format=args.output_format,
    )
