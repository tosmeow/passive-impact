"""Plot utilities for queue-only counterfactual experiments."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    return 'images_without_us' if counterfactual else 'images'


def load_data(mode='single', data_mode='efficient', counterfactual=False, data_base=None):
    base = data_base or f'./data/{mode}/{data_mode}'
    times = np.load(f'{base}/times.npy')
    queue = np.load(f'{base}/queue_paths.npy')
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
    include_title=True,
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
    if include_title:
        ax.set_title(layout['title'])
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {save_path}')
    else:
        plt.show()


def generate_all_plots(
    mode='single',
    data_mode='efficient',
    meta_end=80.0,
    counterfactual=False,
    data_base=None,
    output_dir=None,
    include_title=True,
):
    df = load_data(
        mode,
        data_mode,
        counterfactual=counterfactual,
        data_base=data_base,
    )
    output_dir = output_dir or _default_output_dir(counterfactual)
    plot_queue_shades(
        df,
        counterfactual=counterfactual,
        meta_end=meta_end,
        save_path=os.path.join(output_dir, f'queue_paths_{mode}.png'),
        include_title=include_title,
    )


def parse_args():
    p = argparse.ArgumentParser(description='Generate queue simulation plots.')
    p.add_argument('--mode', default='single', choices=['single', 'double'])
    p.add_argument('--data-mode', default='efficient')
    p.add_argument('--meta-end', type=float, default=80.0)
    p.add_argument('--counterfactual', action='store_true',
                   help='Interpret queue_paths.npy as without-us output: first column bar_q, simulations q.')
    p.add_argument('--data-base', default=None,
                   help='Directory containing times.npy and queue_paths.npy.')
    p.add_argument('--output-dir', default=None,
                   help='Directory where images should be written.')
    p.add_argument('--no-title', action='store_true',
                   help='Do not draw titles on generated PNG images.')
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
        include_title=not args.no_title,
    )
