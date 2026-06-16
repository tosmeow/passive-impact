"""Plot utilities for aggressive impact experiments (propagator + hybrid).

Loads pre-saved baseline data from:
    ./data/propagator/   — pure-propagator model
    ./data/hybrid/       — hybrid propagator+instantaneous model

Usage:
    python plot_utils.py
    python plot_utils.py --model propagator
    python plot_utils.py --model hybrid
    python plot_utils.py --model propagator --scenario without
    python plot_utils.py --title
"""
import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.plot_utils_propagator import (
        compute_plot_y_lims as compute_propagator_y_lims,
        load_data as _load_propagator_data,
        plot_impact_by_event_type,
        plot_queue_diff,
        plot_shades,
    )
    from utils.plot_utils_propagator import generate_all_plots as gen_propagator
    from utils.plot_utils_hybrid import compute_plot_y_lims as compute_hybrid_y_lims
    from utils.plot_utils_hybrid import generate_all_plots as gen_hybrid
    from experiments.plot_utils_common import add_format_argument, add_title_argument
else:
    from .utils.plot_utils_propagator import (
        compute_plot_y_lims as compute_propagator_y_lims,
        load_data as _load_propagator_data,
        plot_impact_by_event_type,
        plot_queue_diff,
        plot_shades,
    )
    from .utils.plot_utils_propagator import generate_all_plots as gen_propagator
    from .utils.plot_utils_hybrid import compute_plot_y_lims as compute_hybrid_y_lims
    from .utils.plot_utils_hybrid import generate_all_plots as gen_hybrid
    from ...plot_utils_common import add_format_argument, add_title_argument


def load_data(counterfactual=False):
    """Load propagator data using the legacy 3-value notebook API."""
    impact_df, queue_df, is_market, _meta_end = _load_propagator_data(
        counterfactual=counterfactual
    )
    return impact_df, queue_df, is_market


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate aggressive impact plots (propagator, hybrid, or both models).'
    )
    p.add_argument('--model', choices=['both', 'propagator', 'hybrid'], default='both',
                   help='Which aggressive impact model results to plot.')
    p.add_argument('--scenario', choices=['with', 'without', 'both'], default=None,
                   help='Which conditioning case to plot (default: both).')
    p.add_argument('--counterfactual', action='store_true',
                   help='Alias for --scenario without: first column bar_q, simulations q.')
    p.add_argument('--data-base', default=None,
                   help='Directory containing saved .npy files. With --model both, expects propagator/ and hybrid/ subdirectories.')
    p.add_argument('--output-dir', default=None,
                   help='Directory where images should be written. With --model both, writes propagator/ and hybrid/ subdirectories.')
    p.add_argument('--bar-kappa', type=float, default=None,
                   help='Hybrid-only value used when data-base does not contain bar_kappa.npy.')
    add_title_argument(p, default=False)
    add_format_argument(p, default='pdf')
    args = p.parse_args()
    if args.counterfactual and args.scenario not in {None, 'without'}:
        p.error('--counterfactual cannot be combined with --scenario with or --scenario both')
    return args


def generate_all_plots(
    model,
    counterfactual=False,
    scenario=None,
    data_base=None,
    output_dir=None,
    bar_kappa=None,
    include_title=False,
    output_format='pdf',
):
    """Generate all plots for the given model.

    Args:
        model: 'both', 'propagator', or 'hybrid'
    """
    if model not in {'both', 'propagator', 'hybrid'}:
        raise ValueError("model must be 'both', 'propagator', or 'hybrid'")

    scenario_counterfactuals = _scenario_counterfactuals(scenario, counterfactual)

    if model in {'both', 'propagator'}:
        propagator_bases = {
            scenario_counterfactual: _model_data_base(
                data_base,
                'propagator',
                model,
                scenario_counterfactual,
            )
            for scenario_counterfactual in scenario_counterfactuals
        }
        propagator_y_lims = _shared_y_lims(
            compute_propagator_y_lims(
                counterfactual=scenario_counterfactual,
                data_base=propagator_bases[scenario_counterfactual],
            )
            for scenario_counterfactual in scenario_counterfactuals
        )
        for scenario_counterfactual in scenario_counterfactuals:
            gen_propagator(
                counterfactual=scenario_counterfactual,
                data_base=propagator_bases[scenario_counterfactual],
                output_dir=_model_output_dir(output_dir, 'propagator', model),
                include_title=include_title,
                y_lims=propagator_y_lims,
                output_format=output_format,
            )

    if model in {'both', 'hybrid'}:
        hybrid_bases = {
            scenario_counterfactual: _model_data_base(
                data_base,
                'hybrid',
                model,
                scenario_counterfactual,
            )
            for scenario_counterfactual in scenario_counterfactuals
        }
        hybrid_y_lims = _shared_y_lims(
            compute_hybrid_y_lims(
                counterfactual=scenario_counterfactual,
                data_base=hybrid_bases[scenario_counterfactual],
                bar_kappa=bar_kappa,
            )
            for scenario_counterfactual in scenario_counterfactuals
        )
        for scenario_counterfactual in scenario_counterfactuals:
            gen_hybrid(
                counterfactual=scenario_counterfactual,
                data_base=hybrid_bases[scenario_counterfactual],
                output_dir=_model_output_dir(output_dir, 'hybrid', model),
                bar_kappa=bar_kappa,
                include_title=include_title,
                y_lims=hybrid_y_lims,
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


def _model_data_base(data_base, model_name, requested_model, counterfactual):
    if data_base is None:
        return None
    path = Path(data_base)
    if requested_model == 'both':
        path = path / model_name
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


def _model_output_dir(output_dir, model_name, requested_model):
    if output_dir is None:
        return None
    path = Path(output_dir)
    if requested_model == 'both':
        return path / model_name
    return path


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(
        args.model,
        counterfactual=args.counterfactual,
        scenario=args.scenario,
        data_base=args.data_base,
        output_dir=args.output_dir,
        bar_kappa=args.bar_kappa,
        include_title=args.include_title,
        output_format=args.output_format,
    )
