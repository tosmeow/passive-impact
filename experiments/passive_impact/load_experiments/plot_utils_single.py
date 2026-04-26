import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Plot single-queue passive impact results.')
    parser.add_argument(
        '--data-mode', choices=['general', 'efficient'], default='efficient',
        help='Which simulation backend results to load (default: efficient).'
    )
    parser.add_argument(
        '--meta-end', type=float, default=80.0,
        help='Time at which the metaorder ends, drawn as a vertical line (default: 80.0).'
    )
    return parser.parse_args()


def load_data(data_mode):
    """Load simulation results from .npy files into pandas DataFrames."""
    data_base = f'../../../data/single_queue/{data_mode}'

    times = np.load(f'{data_base}/with/times.npy')
    times_without = np.load(f'{data_base}/without/times.npy')

    impact_with = np.load(f'{data_base}/with/impact_paths.npy')
    impact_without = np.load(f'{data_base}/without/impact_paths.npy')

    queue_with = np.load(f'{data_base}/with/queue_paths.npy')
    queue_without = np.load(f'{data_base}/without/queue_paths.npy')

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


def plot_queue_shades(df, sim_col, title, label, meta_end, ref_col=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in df.columns if col.startswith(sim_col)]

    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.05, linewidth=0.5)

    avg_sims = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg_sims, color='red', linewidth=2.5, label='Mean')

    if ref_col is not None:
        ax.plot(df.index, df[ref_col], color='black', linewidth=2.5, label=ref_col)

    ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def generate_all_plots(data_mode, meta_end):
    """Generate and save all four analysis plots."""

    path_with, path_without, queue_with, queue_without = load_data(data_mode)

    os.makedirs('images', exist_ok=True)

    print("\nGenerating plots...")

    plot_queue_shades(
        path_with,
        sim_col='sim_',
        title='Conditional Impact I(t) given base queue q',
        label='Price Impact',
        meta_end=meta_end,
        ref_col=None,
        save_path='images/impact_given_q.png'
    )

    plot_queue_shades(
        queue_with,
        sim_col='bar_q',
        title='Counterfactual queue size given base queue q',
        label='Queue Size',
        meta_end=meta_end,
        ref_col='q',
        save_path='images/queue_given_q.png'
    )

    plot_queue_shades(
        path_without,
        sim_col='sim_',
        title='Conditional Impact I(t) given impacted queue q\u0304',
        label='Price Impact',
        meta_end=meta_end,
        ref_col=None,
        save_path='images/impact_given_qbar.png'
    )

    plot_queue_shades(
        queue_without,
        sim_col='q_sim',
        title='Counterfactual queue size given impacted queue q\u0304',
        label='Queue Size',
        meta_end=meta_end,
        ref_col='bar_q',
        save_path='images/queue_given_qbar.png'
    )


if __name__ == '__main__':
    args = parse_args()
    generate_all_plots(args.data_mode, args.meta_end)
