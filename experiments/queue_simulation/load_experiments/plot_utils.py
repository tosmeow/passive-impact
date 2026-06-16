"""Plot utilities for queue-only counterfactual experiments."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiments.plot_utils_common import (
    add_title_argument,
    maybe_set_title,
    save_or_show,
    script_dir,
)

SCRIPT_DIR = script_dir(__file__)


def _queue_layout(counterfactual=False):
    if counterfactual:
        return {
            'ref_col': 'bar_q',
            'sim_prefix': 'q_sim_',
            'columns': lambda n: ['bar_q'] + [f'q_sim_{i}' for i in range(n)],
            'mean_label': 'Mean q',
            'ref_label': '$\\bar{q}$ (reference)',
            'title': 'Queue paths: $\\bar{q}$ reference, q without meta orders',
        }
    return {
        'ref_col': 'q',
        'sim_prefix': 'bar_q_sim_',
        'columns': lambda n: ['q'] + [f'bar_q_sim_{i}' for i in range(n)],
        'mean_label': 'Mean $\\bar{q}$',
        'ref_label': 'q (baseline)',
        'title': 'Counterfactual queue paths under metaorder',
    }


def _default_output_dir(counterfactual):
    dirname = 'images_without_us' if counterfactual else 'images'
    return Path(SCRIPT_DIR) / dirname


def load_data(mode='single', data_mode='efficient', counterfactual=False, data_base=None):
    if data_base is None:
        base_root = Path(SCRIPT_DIR) / 'data' / mode / data_mode
        scenario = 'without' if counterfactual else 'with'
        scenario_base = base_root / scenario
        if (
            (scenario_base / 'times.npy').exists()
            and (scenario_base / 'queue_paths.npy').exists()
        ):
            base = scenario_base
        else:
            base = base_root
    else:
        base = data_base
    times = np.load(Path(base) / 'times.npy')
    queue = np.load(Path(base) / 'queue_paths.npy')
    n_sims = queue.shape[1] - 1
    layout = _queue_layout(counterfactual)
    df = pd.DataFrame(
        queue,
        index=pd.Index(times, name='time'),
        columns=layout['columns'](n_sims),
    )
    return df


def plot_queue_shades(
    df,
    counterfactual=False,
    meta_end=None,
    save_path=None,
    include_title=False,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    layout = _queue_layout(counterfactual)
    sim_cols = [c for c in df.columns if c.startswith(layout['sim_prefix'])]
    if not sim_cols:
        raise ValueError(f"No simulation columns found with prefix {layout['sim_prefix']!r}")
    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.05, linewidth=0.5)
    avg = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg, color='red', linewidth=2.5, label=layout['mean_label'])
    ax.plot(df.index, df[layout['ref_col']], color='black', linewidth=2.5, label=layout['ref_label'])
    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Queue size')
    maybe_set_title(ax, layout['title'], include_title)
    ax.legend()
    plt.tight_layout()
    save_or_show(fig, save_path, dpi=300)


def generate_all_plots(
    mode='single',
    data_mode='efficient',
    meta_end=60.0,
    counterfactual=False,
    data_base=None,
    output_dir=None,
    include_title=False,
):
    df = load_data(
        mode,
        data_mode,
        counterfactual=counterfactual,
        data_base=data_base,
    )
    output_dir = Path(output_dir) if output_dir is not None else _default_output_dir(counterfactual)
    plot_queue_shades(
        df,
        counterfactual=counterfactual,
        meta_end=meta_end,
        save_path=output_dir / f'queue_paths_{mode}.png',
        include_title=include_title,
    )


def parse_args():
    p = argparse.ArgumentParser(description='Generate queue simulation plots.')
    p.add_argument('--mode', default='single', choices=['single', 'double'])
    p.add_argument('--data-mode', default='efficient')
    p.add_argument('--meta-end', type=float, default=60.0)
    p.add_argument('--counterfactual', action='store_true',
                   help='Interpret queue_paths.npy as without-us output: first column bar_q, simulations q.')
    p.add_argument('--data-base', default=None,
                   help='Directory containing times.npy and queue_paths.npy.')
    p.add_argument('--output-dir', default=None,
                   help='Directory where images should be written.')
    add_title_argument(p, default=False)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(
        mode=args.mode,
        data_mode=args.data_mode,
        meta_end=args.meta_end,
        counterfactual=args.counterfactual,
        data_base=args.data_base,
        output_dir=args.output_dir,
        include_title=args.include_title,
    )
