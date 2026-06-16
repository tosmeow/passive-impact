import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiments.plot_utils_common import (
    add_format_argument,
    add_title_argument,
    data_dir,
    image_dir,
    maybe_set_title,
    save_or_show,
    with_output_format,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot single-queue passive impact results.')
    parser.add_argument(
        '--data-mode', choices=['general', 'efficient'], default='efficient',
        help='Which simulation backend results to load (default: efficient).'
    )
    parser.add_argument(
        '--meta-end', type=float, default=60.0,
        help='Time at which the metaorder ends, drawn as a vertical line (default: 60.0).'
    )
    add_title_argument(parser, default=False)
    add_format_argument(parser, default='png')
    return parser.parse_args()


def load_data(data_mode):
    """Load simulation results from .npy files into pandas DataFrames."""
    data_base = data_dir(__file__, 'single', data_mode)

    times = np.load(data_base / 'with' / 'times.npy')
    times_without = np.load(data_base / 'without' / 'times.npy')

    impact_with = np.load(data_base / 'with' / 'impact_paths.npy')
    impact_without = np.load(data_base / 'without' / 'impact_paths.npy')

    queue_with = np.load(data_base / 'with' / 'queue_paths.npy')
    queue_without = np.load(data_base / 'without' / 'queue_paths.npy')

    n_sims_with = impact_with.shape[1]
    n_sims_without = impact_without.shape[1]

    path_with = pd.DataFrame(
        impact_with,
        index=pd.Index(times, name='time'),
        columns=[f'sim_{i}' for i in range(n_sims_with)]
    )

    path_without = pd.DataFrame(
        impact_without,
        index=pd.Index(times_without, name='time'),
        columns=[f'sim_{i}' for i in range(n_sims_without)]
    )

    # Queue with: first col is 'q', rest are 'bar_q_sim_i'
    queue_with_df = pd.DataFrame(
        queue_with,
        index=pd.Index(times, name='time'),
        columns=['q'] + [f'bar_q_sim_{i}' for i in range(n_sims_with)]
    )

    # Queue without: first col is 'bar_q', rest are 'q_sim_i'
    queue_without_df = pd.DataFrame(
        queue_without,
        index=pd.Index(times_without, name='time'),
        columns=['bar_q'] + [f'q_sim_{i}' for i in range(n_sims_without)]
    )

    return path_with, path_without, queue_with_df, queue_without_df


def _default_mean_label(sim_col, ylabel):
    if 'impact' in ylabel.lower():
        return 'Mean impact'
    if sim_col.startswith('q_sim'):
        return 'Mean q'
    if sim_col.startswith('bar_q'):
        return 'Mean $\\bar{q}$'
    return 'Mean'


def _display_col_label(col):
    if col == 'bar_q':
        return '$\\bar{q}$'
    return col


def plot_queue_shades(
    df,
    sim_col,
    title,
    label,
    meta_end=None,
    ref_col=None,
    save_path=None,
    mean_label=None,
    include_title=False,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in df.columns if col.startswith(sim_col)]

    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.05, linewidth=0.5)

    if mean_label is None:
        mean_label = _default_mean_label(sim_col, label)

    avg_sims = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg_sims, color='red', linewidth=2.5, label=mean_label)

    if ref_col is not None:
        ax.plot(
            df.index,
            df[ref_col],
            color='black',
            linewidth=2.5,
            label=_display_col_label(ref_col),
        )

    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(label)
    maybe_set_title(ax, title, include_title)
    ax.legend()
    plt.tight_layout()

    save_or_show(fig, save_path, dpi=300)


def generate_all_plots(data_mode, meta_end, include_title=False, output_format='png'):
    """Generate and save all four analysis plots."""

    path_with, path_without, queue_with, queue_without = load_data(data_mode)

    output_dir = image_dir(__file__)

    print("\nGenerating plots...")

    plot_queue_shades(
        path_with,
        sim_col='sim_',
        title='Conditional Impact I(t) given base queue q',
        label='Price Impact',
        meta_end=meta_end,
        ref_col=None,
        save_path=with_output_format(output_dir / 'impact_given_q.png', output_format),
        mean_label='Mean impact',
        include_title=include_title,
    )

    plot_queue_shades(
        queue_with,
        sim_col='bar_q',
        title='Counterfactual queue size given base queue q',
        label='Queue Size',
        meta_end=meta_end,
        ref_col='q',
        save_path=with_output_format(output_dir / 'queue_given_q.png', output_format),
        mean_label='Mean $\\bar{q}$',
        include_title=include_title,
    )

    plot_queue_shades(
        path_without,
        sim_col='sim_',
        title='Conditional Impact I(t) given impacted queue q\u0304',
        label='Price Impact',
        meta_end=meta_end,
        ref_col=None,
        save_path=with_output_format(output_dir / 'impact_given_qbar.png', output_format),
        mean_label='Mean impact',
        include_title=include_title,
    )

    plot_queue_shades(
        queue_without,
        sim_col='q_sim',
        title='Counterfactual queue size given impacted queue q\u0304',
        label='Queue Size',
        meta_end=meta_end,
        ref_col='bar_q',
        save_path=with_output_format(output_dir / 'queue_given_qbar.png', output_format),
        mean_label='Mean q',
        include_title=include_title,
    )


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(
        args.data_mode,
        args.meta_end,
        include_title=args.include_title,
        output_format=args.output_format,
    )
