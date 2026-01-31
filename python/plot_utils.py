import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def load_data():
    """Load simulation results from .npy files into pandas DataFrames."""
    # Load times
    times = np.load('../times.npy')
    times_without = np.load('../times_without.npy')

    # Load impact paths (n_times x n_simulations)
    impact_with = np.load('../impact_paths.npy')
    impact_without = np.load('../impact_paths_without.npy')

    # Load queue paths (n_times x (1 + n_simulations))
    # First column is reference (q or bar_q), rest are simulations
    queue_with = np.load('../queue_paths.npy')
    queue_without = np.load('../queue_paths_without.npy')

    n_sims_with = impact_with.shape[1]
    n_sims_without = impact_without.shape[1]

    # Build DataFrames matching original structure
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


def plot_queue_shades(df, sim_col, title, label, ref_col=None, save_path=None):
    """Plot simulation paths with shaded individual trajectories and mean.

    Args:
        df: DataFrame with time index and simulation columns
        sim_col: Prefix for simulation columns (e.g., 'sim_', 'bar_q_sim_')
        title: Plot title
        label: Y-axis label
        ref_col: Optional reference column to plot in black
        save_path: If provided, save figure to this path instead of showing
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in df.columns if col.startswith(sim_col)]

    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.1, linewidth=0.5)

    avg_sims = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg_sims, color='red', linewidth=2.5, label=f'Mean')

    if ref_col is not None:
        ax.plot(df.index, df[ref_col], color='black', linewidth=2.5, label=ref_col)

    ax.set_xlabel('Time')
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def generate_all_plots():
    """Generate and save all four analysis plots."""
    print("Loading data...")
    path_with, path_without, queue_with, queue_without = load_data()

    print(f"Data loaded: {path_with.shape[1]} simulations, {len(path_with)} time points")

    # Create output directory if needed
    os.makedirs('.', exist_ok=True)

    print("\nGenerating plots...")

    # 1. Impact given q (paths_with_us scenario)
    plot_queue_shades(
        path_with,
        sim_col='sim_',
        title='Conditional Impact I(t) given baseline q',
        label='Price Impact',
        ref_col=None,
        save_path='impact_given_q.png'
    )

    # 2. Queue bar_q given q
    plot_queue_shades(
        queue_with,
        sim_col='bar_q',
        title='Counterfactual queue size given baseline q',
        label='Queue Size',
        ref_col='q',
        save_path='queue_given_q.png'
    )

    # 3. Impact given bar_q (paths_without_us scenario)
    plot_queue_shades(
        path_without,
        sim_col='sim_',
        title='Conditional Impact I(t) given baseline q̄',
        label='Price Impact',
        ref_col=None,
        save_path='impact_given_qbar.png'
    )

    # 4. Queue q given bar_q
    plot_queue_shades(
        queue_without,
        sim_col='q_sim',
        title='Counterfactual queue size given baseline q̄',
        label='Queue Size',
        ref_col='bar_q',
        save_path='queue_given_qbar.png'
    )

    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    generate_all_plots()
