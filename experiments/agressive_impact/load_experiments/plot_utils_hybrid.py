import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

DATA_BASE = './data/hybrid'


def load_data():
    """Load hybrid aggressive impact simulation results from .npy files."""
    times = np.load(f'{DATA_BASE}/times.npy')
    impact_paths = np.load(f'{DATA_BASE}/impact_paths.npy')
    queue_paths = np.load(f'{DATA_BASE}/queue_paths.npy')
    event_types = np.load(f'{DATA_BASE}/event_types.npy')
    bar_kappa = float(np.load(f'{DATA_BASE}/bar_kappa.npy')[0])

    n_sims = impact_paths.shape[1]
    is_market = event_types == 1.0

    impact_df = pd.DataFrame(
        impact_paths,
        index=pd.Index(times, name='time'),
        columns=[f'sim_{i}' for i in range(n_sims)]
    )

    queue_df = pd.DataFrame(
        queue_paths,
        index=pd.Index(times, name='time'),
        columns=['q'] + [f'bar_q_sim_{i}' for i in range(n_sims)]
    )

    meta_mask = ~is_market
    meta_end = times[meta_mask].max() if meta_mask.any() else None

    return impact_df, queue_df, is_market, meta_end, bar_kappa


def plot_shades(df, sim_prefix, title, ylabel, meta_end=None, ref_col=None, save_path=None):
    """Plot individual simulation paths as transparent lines with mean overlay."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in df.columns if col.startswith(sim_prefix)]

    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.05, linewidth=0.5)

    avg = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg, color='red', linewidth=2.5, label='Mean')

    if ref_col is not None and ref_col in df.columns:
        ax.plot(df.index, df[ref_col], color='black', linewidth=2.5, label=ref_col)

    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_impact_decomposition(impact_df, is_market, bar_kappa, meta_end=None, save_path=None):
    """Plot mean impact at market order vs metaorder times to show the two contributions."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in impact_df.columns if col.startswith('sim_')]
    mean_impact = impact_df[sim_cols].mean(axis=1)

    times = impact_df.index.values
    market_mask = is_market
    meta_mask = ~is_market

    ax.scatter(times[market_mask], mean_impact.values[market_mask],
               s=1, alpha=0.5, color='blue',
               label=r'At market orders (instantaneous: $\kappa(\bar{q}) - \kappa(q)$)')
    ax.scatter(times[meta_mask], mean_impact.values[meta_mask],
               s=3, alpha=0.7, color='red',
               label=rf'At meta orders (propagator: $\bar{{\kappa}} G(t-s)$, $\bar{{\kappa}}={bar_kappa:.1f}$)')

    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Mean Impact MI(t)')
    ax.set_title('Hybrid Aggressive Impact by Event Type')
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_queue_diff(queue_df, meta_end=None, save_path=None):
    """Plot the queue difference bar_q - q over time with quantile bands."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in queue_df.columns if col.startswith('bar_q_sim_')]
    q = queue_df['q'].values

    diffs = queue_df[sim_cols].astype(np.int64).subtract(q.astype(np.int64), axis=0)
    mean_diff = diffs.mean(axis=1)
    q10 = diffs.quantile(0.10, axis=1)
    q25 = diffs.quantile(0.25, axis=1)
    q75 = diffs.quantile(0.75, axis=1)
    q90 = diffs.quantile(0.90, axis=1)

    times = diffs.index.values

    ax.fill_between(times, q10, q90, alpha=0.15, color='blue', label='10%\u201390%')
    ax.fill_between(times, q25, q75, alpha=0.3, color='blue', label='25%\u201375%')
    ax.plot(times, mean_diff, color='red', linewidth=2.5, label='Mean $\\bar{q} - q$')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Queue difference $\\bar{q} - q$')
    ax.set_title('Queue depletion from aggressive meta orders')
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def generate_all_plots():
    """Generate and save all analysis plots."""
    impact_df, queue_df, is_market, meta_end, bar_kappa = load_data()

    os.makedirs('images', exist_ok=True)

    print(f"Generating plots (bar_kappa={bar_kappa:.4f}, metaorder ends at t={meta_end:.2f})...")

    plot_shades(
        impact_df,
        sim_prefix='sim_',
        title=r'Aggressive Market Impact MI(t)',
        ylabel='Price Impact',
        meta_end=meta_end,
        save_path='images/impact_paths.png'
    )

    plot_shades(
        queue_df,
        sim_prefix='bar_q_sim_',
        title='Queue dynamics: q (reference) vs $\\bar{q}$ (with meta orders)',
        ylabel='Queue Size',
        meta_end=meta_end,
        ref_col='q',
        save_path='images/queue_paths.png'
    )

    plot_impact_decomposition(
        impact_df,
        is_market,
        bar_kappa,
        meta_end=meta_end,
        save_path='images/impact_by_event_type.png'
    )

    plot_queue_diff(
        queue_df,
        meta_end=meta_end,
        save_path='images/queue_diff.png'
    )


if __name__ == '__main__':
    generate_all_plots()
