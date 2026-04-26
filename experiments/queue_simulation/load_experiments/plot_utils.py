"""Plot utilities for queue-only counterfactual experiments."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(mode='single', data_mode='efficient'):
    base = f'./data/{mode}/{data_mode}'
    times = np.load(f'{base}/times.npy')
    queue = np.load(f'{base}/queue_paths.npy')
    n_sims = queue.shape[1] - 1
    df = pd.DataFrame(
        queue,
        index=pd.Index(times, name='time'),
        columns=['q'] + [f'bar_q_sim_{i}' for i in range(n_sims)],
    )
    return df


def plot_queue_shades(df, meta_end=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sim_cols = [c for c in df.columns if c.startswith('bar_q_sim_')]
    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.05, linewidth=0.5)
    avg = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg, color='red', linewidth=2.5, label='Mean $\\bar{q}$')
    ax.plot(df.index, df['q'], color='black', linewidth=2.5, label='q (baseline)')
    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Queue size')
    ax.set_title('Counterfactual queue paths under metaorder')
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {save_path}')
    else:
        plt.show()


def generate_all_plots(mode='single', data_mode='efficient', meta_end=80.0):
    df = load_data(mode, data_mode)
    plot_queue_shades(df, meta_end=meta_end, save_path=f'images/queue_paths_{mode}.png')


if __name__ == '__main__':
    generate_all_plots()
