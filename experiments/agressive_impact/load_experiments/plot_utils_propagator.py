import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

DATA_BASE = './data/propagator'


def _queue_layout(counterfactual=False):
    if counterfactual:
        return {
            'ref_col': 'bar_q',
            'sim_prefix': 'q_sim_',
            'columns': lambda n: ['bar_q'] + [f'q_sim_{i}' for i in range(n)],
            'mean_label': 'Mean q',
            'queue_title': 'Queue dynamics: $\\bar{q}$ (reference) vs q (without meta orders)',
        }
    return {
        'ref_col': 'q',
        'sim_prefix': 'bar_q_sim_',
        'columns': lambda n: ['q'] + [f'bar_q_sim_{i}' for i in range(n)],
        'mean_label': 'Mean $\\bar{q}$',
        'queue_title': 'Queue dynamics: q (reference) vs $\\bar{q}$ (with meta orders)',
    }


def _default_output_dir(counterfactual):
    return 'images_without_us' if counterfactual else 'images'


def _queue_diffs(queue_df, counterfactual=False):
    """Return bar_q - q for each simulation path under either direction."""
    layout = _queue_layout(counterfactual)
    if layout['ref_col'] not in queue_df.columns:
        raise ValueError(f"Missing reference column {layout['ref_col']!r}")
    sim_cols = [col for col in queue_df.columns if col.startswith(layout['sim_prefix'])]
    if not sim_cols:
        raise ValueError(f"No simulation columns found with prefix {layout['sim_prefix']!r}")

    reference = queue_df[layout['ref_col']].astype(np.int64).to_numpy()
    simulated = queue_df[sim_cols].astype(np.int64).to_numpy()
    if counterfactual:
        values = reference[:, None] - simulated
    else:
        values = simulated - reference[:, None]
    return pd.DataFrame(values, index=queue_df.index, columns=sim_cols)


def load_data(counterfactual=False, data_base=None):
    """Load aggressive impact simulation results from .npy files."""
    data_base = data_base or DATA_BASE
    times = np.load(f'{data_base}/times.npy')
    impact_paths = np.load(f'{data_base}/impact_paths.npy')
    queue_paths = np.load(f'{data_base}/queue_paths.npy')
    event_types = np.load(f'{data_base}/event_types.npy')

    n_sims = impact_paths.shape[1]
    is_market = event_types == 1.0

    impact_df = pd.DataFrame(
        impact_paths,
        index=pd.Index(times, name='time'),
        columns=[f'sim_{i}' for i in range(n_sims)]
    )

    layout = _queue_layout(counterfactual)
    queue_df = pd.DataFrame(
        queue_paths,
        index=pd.Index(times, name='time'),
        columns=layout['columns'](n_sims)
    )

    # Derive metaorder end from the last metaorder event time (event_type == 0)
    meta_mask = ~is_market
    meta_end = times[meta_mask].max() if meta_mask.any() else None

    return impact_df, queue_df, is_market, meta_end


def plot_shades(
    df,
    sim_prefix,
    title,
    ylabel,
    meta_end=None,
    ref_col=None,
    save_path=None,
    mean_label='Mean',
):
    """Plot individual simulation paths as transparent lines with mean overlay."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in df.columns if col.startswith(sim_prefix)]

    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.05, linewidth=0.5)

    if not sim_cols:
        raise ValueError(f"No simulation columns found with prefix {sim_prefix!r}")

    avg = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg, color='red', linewidth=2.5, label=mean_label)

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


def plot_impact_by_event_type(impact_df, is_market, meta_end=None, save_path=None):
    """Plot mean impact separately at market order and meta order times."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in impact_df.columns if col.startswith('sim_')]
    mean_impact = impact_df[sim_cols].mean(axis=1)

    times = impact_df.index.values

    market_mask = is_market
    meta_mask = ~is_market

    ax.scatter(times[market_mask], mean_impact.values[market_mask],
               s=1, alpha=0.5, color='blue', label='At market orders (N)')
    ax.scatter(times[meta_mask], mean_impact.values[meta_mask],
               s=3, alpha=0.7, color='red', label='At meta orders (N$^o$)')

    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Mean Impact MI(t)')
    ax.set_title('Aggressive Impact by Event Type')
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_queue_diff(queue_df, counterfactual=False, meta_end=None, save_path=None):
    """Plot the queue difference bar_q - q over time with quantile bands."""
    fig, ax = plt.subplots(figsize=(12, 6))

    diffs = _queue_diffs(queue_df, counterfactual=counterfactual)
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


def generate_all_plots(counterfactual=False, data_base=None, output_dir=None):
    """Generate and save all analysis plots."""
    impact_df, queue_df, is_market, meta_end = load_data(
        counterfactual=counterfactual,
        data_base=data_base,
    )
    layout = _queue_layout(counterfactual)

    output_dir = output_dir or _default_output_dir(counterfactual)
    os.makedirs(output_dir, exist_ok=True)

    direction = 'without us' if counterfactual else 'with us'
    meta_msg = f"{meta_end:.2f}" if meta_end is not None else "unknown"
    print(f"Generating {direction} plots (metaorder ends at t={meta_msg})...")

    plot_shades(
        impact_df,
        sim_prefix='sim_',
        title='Aggressive Market Impact MI(t)',
        ylabel='Price Impact',
        meta_end=meta_end,
        save_path=os.path.join(output_dir, 'impact_paths.png')
    )

    plot_shades(
        queue_df,
        sim_prefix=layout['sim_prefix'],
        title=layout['queue_title'],
        ylabel='Queue Size',
        meta_end=meta_end,
        ref_col=layout['ref_col'],
        save_path=os.path.join(output_dir, 'queue_paths.png'),
        mean_label=layout['mean_label'],
    )

    plot_impact_by_event_type(
        impact_df,
        is_market,
        meta_end=meta_end,
        save_path=os.path.join(output_dir, 'impact_by_event_type.png')
    )

    plot_queue_diff(
        queue_df,
        counterfactual=counterfactual,
        meta_end=meta_end,
        save_path=os.path.join(output_dir, 'queue_diff.png')
    )


if __name__ == '__main__':
    generate_all_plots()
