import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

# Compact color scheme
COLORS = {'ask': '#2563eb', 'bid': '#ea580c', 'impact': '#16a34a', 'sim': '#9ca3af'}

# Data directory - can be 'general' or 'efficient'
DATA_MODE = 'efficient'  # Change to 'general' for non-memory-efficient data
DATA_BASE = f'../../../data/double_queue/{DATA_MODE}'


def load_bidask_data():
    """Load bid-ask simulation results from .npy files into pandas DataFrames.

    Returns:
        dict with keys:
            - 'ask_impact_with', 'ask_impact_without': Impact DataFrames for ask side
            - 'bid_impact_with', 'bid_impact_without': Impact DataFrames for bid side
            - 'ask_queue_with', 'ask_queue_without': Ask queue DataFrames
            - 'bid_queue_with', 'bid_queue_without': Bid queue DataFrames
            - 'ask_times', 'bid_times': Time arrays
            - 'ask_times_without', 'bid_times_without': Time arrays for without scenario
    """
    # Load times
    ask_times = np.load(f'{DATA_BASE}/with/ask_times.npy')
    bid_times = np.load(f'{DATA_BASE}/with/bid_times.npy')
    ask_times_without = np.load(f'{DATA_BASE}/without/ask_times.npy')
    bid_times_without = np.load(f'{DATA_BASE}/without/bid_times.npy')

    # Load impact paths (n_times x n_simulations)
    ask_impact_with = np.load(f'{DATA_BASE}/with/ask_impact_paths.npy')
    ask_impact_without = np.load(f'{DATA_BASE}/without/ask_impact_paths.npy')
    bid_impact_with = np.load(f'{DATA_BASE}/with/bid_impact_paths.npy')
    bid_impact_without = np.load(f'{DATA_BASE}/without/bid_impact_paths.npy')

    # Load queue paths (n_times x (1 + n_simulations))
    ask_queue_with = np.load(f'{DATA_BASE}/with/ask_queue_paths.npy')
    ask_queue_without = np.load(f'{DATA_BASE}/without/ask_queue_paths.npy')
    bid_queue_with = np.load(f'{DATA_BASE}/with/bid_queue_paths.npy')
    bid_queue_without = np.load(f'{DATA_BASE}/without/bid_queue_paths.npy')

    n_sims_with = ask_impact_with.shape[1]
    n_sims_without = ask_impact_without.shape[1]

    # Build impact DataFrames
    ask_impact_with_df = pd.DataFrame(
        ask_impact_with,
        index=pd.Index(ask_times, name='time'),
        columns=[f'sim_{i}' for i in range(n_sims_with)]
    )
    ask_impact_without_df = pd.DataFrame(
        ask_impact_without,
        index=pd.Index(ask_times_without, name='time'),
        columns=[f'sim_{i}' for i in range(n_sims_without)]
    )
    bid_impact_with_df = pd.DataFrame(
        bid_impact_with,
        index=pd.Index(bid_times, name='time'),
        columns=[f'sim_{i}' for i in range(n_sims_with)]
    )
    bid_impact_without_df = pd.DataFrame(
        bid_impact_without,
        index=pd.Index(bid_times_without, name='time'),
        columns=[f'sim_{i}' for i in range(n_sims_without)]
    )

    # Queue with: first col is reference (q), rest are counterfactual (bar_q)
    ask_queue_with_df = pd.DataFrame(
        ask_queue_with,
        index=pd.Index(ask_times, name='time'),
        columns=['q_a'] + [f'bar_q_a_sim_{i}' for i in range(n_sims_with)]
    )
    bid_queue_with_df = pd.DataFrame(
        bid_queue_with,
        index=pd.Index(bid_times, name='time'),
        columns=['q_b'] + [f'bar_q_b_sim_{i}' for i in range(n_sims_with)]
    )

    # Queue without: first col is reference (bar_q), rest are counterfactual (q)
    ask_queue_without_df = pd.DataFrame(
        ask_queue_without,
        index=pd.Index(ask_times_without, name='time'),
        columns=['bar_q_a'] + [f'q_a_sim_{i}' for i in range(n_sims_without)]
    )
    bid_queue_without_df = pd.DataFrame(
        bid_queue_without,
        index=pd.Index(bid_times_without, name='time'),
        columns=['bar_q_b'] + [f'q_b_sim_{i}' for i in range(n_sims_without)]
    )

    return {
        'ask_impact_with': ask_impact_with_df,
        'ask_impact_without': ask_impact_without_df,
        'bid_impact_with': bid_impact_with_df,
        'bid_impact_without': bid_impact_without_df,
        'ask_queue_with': ask_queue_with_df,
        'ask_queue_without': ask_queue_without_df,
        'bid_queue_with': bid_queue_with_df,
        'bid_queue_without': bid_queue_without_df,
        'ask_times': ask_times,
        'bid_times': bid_times,
        'ask_times_without': ask_times_without,
        'bid_times_without': bid_times_without,
    }


def plot_dual_queue_shades(ask_df, bid_df, ask_sim_col, bid_sim_col,
                           title, ask_ref_col=None, bid_ref_col=None,
                           save_path=None):
    """Plot both ask and bid queues on the same graph with shaded simulations.

    Args:
        ask_df: DataFrame for ask queue
        bid_df: DataFrame for bid queue
        ask_sim_col: Prefix for ask simulation columns
        bid_sim_col: Prefix for bid simulation columns
        title: Plot title
        ask_ref_col: Reference column name for ask (e.g., 'q_a')
        bid_ref_col: Reference column name for bid (e.g., 'q_b')
        save_path: Path to save figure (None to show)
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Ask simulations (blue shades)
    ask_sim_cols = [col for col in ask_df.columns if col.startswith(ask_sim_col)]
    for col in ask_sim_cols:
        ax.plot(ask_df.index, ask_df[col], color='blue', alpha=0.05, linewidth=0.5)

    # Bid simulations (orange shades)
    bid_sim_cols = [col for col in bid_df.columns if col.startswith(bid_sim_col)]
    for col in bid_sim_cols:
        ax.plot(bid_df.index, bid_df[col], color='orange', alpha=0.05, linewidth=0.5)

    # Ask mean
    ask_avg = ask_df[ask_sim_cols].mean(axis=1)
    ax.plot(ask_df.index, ask_avg, color='blue', linewidth=2.5,
            label=f'Ask counterfactual mean', linestyle='--')

    # Bid mean
    bid_avg = bid_df[bid_sim_cols].mean(axis=1)
    ax.plot(bid_df.index, bid_avg, color='orange', linewidth=2.5,
            label=f'Bid counterfactual mean', linestyle='--')

    # Reference lines
    if ask_ref_col is not None and ask_ref_col in ask_df.columns:
        ax.plot(ask_df.index, ask_df[ask_ref_col], color='darkblue',
                linewidth=2.5, label=f'Ask reference ({ask_ref_col})')

    if bid_ref_col is not None and bid_ref_col in bid_df.columns:
        ax.plot(bid_df.index, bid_df[bid_ref_col], color='darkorange',
                linewidth=2.5, label=f'Bid reference ({bid_ref_col})')

    ax.set_xlabel('Time')
    ax.set_ylabel('Queue Size')
    ax.set_title(title)
    ax.legend(loc='best')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_impact_shades(df, title, label='Price Impact', color='green',
                       save_path=None):
    """Plot impact paths with shaded simulations.

    Args:
        df: DataFrame with simulation columns named 'sim_*'
        title: Plot title
        label: Y-axis label
        color: Color for mean line
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in df.columns if col.startswith('sim_')]

    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.1, linewidth=0.5)

    avg_sims = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg_sims, color=color, linewidth=2.5, label='Mean')

    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
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


def plot_triple_impact(ask_df, bid_df, title_prefix, scenario='with',
                       save_path=None):
    """Plot the merged price impact (ask - bid) interpolated onto common grid.

    The price impact is ask - bid:
        Total Impact = Σ (q'^a - q^a) at N^a - Σ (q'^b - q^b) at N^b + tails

    This function interpolates both ask and bid impacts onto the union of their
    time grids, computes the difference for each simulation, and plots with
    shaded simulations and mean ± std bands.

    Args:
        ask_df: Ask impact DataFrame
        bid_df: Bid impact DataFrame
        title_prefix: Prefix for title
        scenario: 'with' or 'without' for labeling
        save_path: Path to save figure
    """
    # Use the merged difference function
    plot_impact_difference(ask_df, bid_df,
                          title=f'{title_prefix} - Total Price Impact (Ask - Bid)',
                          save_path=save_path)


def compute_impact_difference(ask_df, bid_df):
    """Compute the difference impact (ask - bid) on a common time grid.

    Interpolates both impacts onto the intersection of time ranges to avoid
    edge artifacts from extrapolation.

    Returns:
        diff_df: DataFrame with columns 'sim_0', 'sim_1', ... for difference
        common_times: array of times
    """
    ask_sim_cols = [col for col in ask_df.columns if col.startswith('sim_')]
    bid_sim_cols = [col for col in bid_df.columns if col.startswith('sim_')]

    n_sims = min(len(ask_sim_cols), len(bid_sim_cols))

    # Use intersection of time ranges to avoid extrapolation artifacts
    t_min = max(ask_df.index.min(), bid_df.index.min())
    t_max = min(ask_df.index.max(), bid_df.index.max())

    all_times = np.sort(np.unique(np.concatenate([ask_df.index.values, bid_df.index.values])))
    common_times = all_times[(all_times >= t_min) & (all_times <= t_max)]

    diff_data = {}
    for i in range(n_sims):
        ask_vals = ask_df[f'sim_{i}'].values
        bid_vals = bid_df[f'sim_{i}'].values

        # Interpolate with last-value extrapolation for safety
        ask_interp = interp1d(ask_df.index, ask_vals, kind='previous',
                              bounds_error=False, fill_value=(ask_vals[0], ask_vals[-1]))
        bid_interp = interp1d(bid_df.index, bid_vals, kind='previous',
                              bounds_error=False, fill_value=(bid_vals[0], bid_vals[-1]))

        diff_data[f'sim_{i}'] = ask_interp(common_times) - bid_interp(common_times)

    diff_df = pd.DataFrame(diff_data, index=pd.Index(common_times, name='time'))
    return diff_df, common_times


def plot_impact_difference(ask_df, bid_df, title, save_path=None):
    """Plot the difference of impacts (Ask - Bid) with all simulations.

    Args:
        ask_df: Ask impact DataFrame
        bid_df: Bid impact DataFrame
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    diff_df, common_times = compute_impact_difference(ask_df, bid_df)
    sim_cols = [col for col in diff_df.columns if col.startswith('sim_')]

    # Plot all simulations
    for col in sim_cols:
        ax.plot(diff_df.index, diff_df[col], color='gray', alpha=0.1, linewidth=0.5)

    # Mean
    diff_mean = diff_df[sim_cols].mean(axis=1)
    ax.plot(diff_df.index, diff_mean, color='green', linewidth=2.5, label='Mean (Ask - Bid)')

    # Std bands
    diff_std = diff_df[sim_cols].std(axis=1)
    ax.fill_between(diff_df.index, diff_mean - diff_std, diff_mean + diff_std,
                    color='green', alpha=0.2, label='±1 std')

    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price Impact (Ask - Bid)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_dashboard(data, save_path=None):
    """3x2 dashboard: queues, individual impacts, and combined impact.

    Layout:
        [Queue given q]              [Queue given q̄]
        [Ask & Bid impact given q]   [Ask & Bid impact given q̄]
        [Impact diff given q]        [Impact diff given q̄]
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    # ===== Row 0: Queues =====
    _plot_queue_panel(axes[0, 0], data['ask_queue_with'], data['bid_queue_with'],
                      sim_prefix_a='bar_q_a', sim_prefix_b='bar_q_b',
                      ref_col_a='q_a', ref_col_b='q_b',
                      title='Queues given base q')

    _plot_queue_panel(axes[0, 1], data['ask_queue_without'], data['bid_queue_without'],
                      sim_prefix_a='q_a_sim', sim_prefix_b='q_b_sim',
                      ref_col_a='bar_q_a', ref_col_b='bar_q_b',
                      title='Queues given impacted q̄')

    # ===== Row 1: Individual Ask & Bid impacts =====
    _plot_individual_impact_panel(axes[1, 0],
                                  data['ask_impact_with'], data['bid_impact_with'],
                                  title='Ask & Bid Impact given base q')

    _plot_individual_impact_panel(axes[1, 1],
                                  data['ask_impact_without'], data['bid_impact_without'],
                                  title='Ask & Bid Impact given impacted q̄')

    # ===== Row 2: Combined impact (Ask - Bid) =====
    _plot_impact_panel(axes[2, 0], data['ask_impact_with'], data['bid_impact_with'],
                       title='Price Impact (Ask − Bid) given base q')

    _plot_impact_panel(axes[2, 1], data['ask_impact_without'], data['bid_impact_without'],
                       title='Price Impact (Ask − Bid) given impacted q̄')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _plot_queue_panel(ax, ask_df, bid_df, sim_prefix_a, sim_prefix_b,
                      ref_col_a, ref_col_b, title):
    """Helper: plot queues on a single axis."""
    # Simulation paths (thin, transparent)
    ask_sim_cols = [c for c in ask_df.columns if c.startswith(sim_prefix_a)]
    bid_sim_cols = [c for c in bid_df.columns if c.startswith(sim_prefix_b)]

    for col in ask_sim_cols[:20]:  # limit to 20 for clarity
        ax.plot(ask_df.index, ask_df[col], color=COLORS['ask'], alpha=0.08, lw=0.5)
    for col in bid_sim_cols[:20]:
        ax.plot(bid_df.index, bid_df[col], color=COLORS['bid'], alpha=0.08, lw=0.5)

    # Means
    ask_mean = ask_df[ask_sim_cols].mean(axis=1)
    bid_mean = bid_df[bid_sim_cols].mean(axis=1)
    ax.plot(ask_df.index, ask_mean, color=COLORS['ask'], lw=2, ls='--', label='Ask counterfactual')
    ax.plot(bid_df.index, bid_mean, color=COLORS['bid'], lw=2, ls='--', label='Bid counterfactual')

    # Reference
    if ref_col_a in ask_df.columns:
        ax.plot(ask_df.index, ask_df[ref_col_a], color=COLORS['ask'], lw=2, label='Ask reference')
    if ref_col_b in bid_df.columns:
        ax.plot(bid_df.index, bid_df[ref_col_b], color=COLORS['bid'], lw=2, label='Bid reference')

    ax.set_xlabel('Time')
    ax.set_ylabel('Queue Size')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)


def _plot_individual_impact_panel(ax, ask_df, bid_df, title):
    """Helper: plot ask and bid impacts separately on a single axis."""
    ask_sim_cols = [c for c in ask_df.columns if c.startswith('sim_')]
    bid_sim_cols = [c for c in bid_df.columns if c.startswith('sim_')]

    # Simulation paths
    for col in ask_sim_cols[:20]:
        ax.plot(ask_df.index, ask_df[col], color=COLORS['ask'], alpha=0.08, lw=0.5)
    for col in bid_sim_cols[:20]:
        ax.plot(bid_df.index, bid_df[col], color=COLORS['bid'], alpha=0.08, lw=0.5)

    # Means
    ask_mean = ask_df[ask_sim_cols].mean(axis=1)
    bid_mean = bid_df[bid_sim_cols].mean(axis=1)
    ax.plot(ask_df.index, ask_mean, color=COLORS['ask'], lw=2.5, label='Ask impact (mean)')
    ax.plot(bid_df.index, bid_mean, color=COLORS['bid'], lw=2.5, label='Bid impact (mean)')

    # Std bands
    ask_std = ask_df[ask_sim_cols].std(axis=1)
    bid_std = bid_df[bid_sim_cols].std(axis=1)
    ax.fill_between(ask_df.index, ask_mean - ask_std, ask_mean + ask_std,
                    color=COLORS['ask'], alpha=0.15)
    ax.fill_between(bid_df.index, bid_mean - bid_std, bid_mean + bid_std,
                    color=COLORS['bid'], alpha=0.15)

    ax.axhline(0, color='black', ls=':', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Impact')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)


def _plot_impact_panel(ax, ask_df, bid_df, title):
    """Helper: plot merged impact (ask - bid) on a single axis."""
    diff_df, _ = compute_impact_difference(ask_df, bid_df)
    sim_cols = [c for c in diff_df.columns if c.startswith('sim_')]

    # Simulation paths
    for col in sim_cols[:20]:
        ax.plot(diff_df.index, diff_df[col], color=COLORS['sim'], alpha=0.15, lw=0.5)

    # Mean and std
    mean = diff_df[sim_cols].mean(axis=1)
    std = diff_df[sim_cols].std(axis=1)
    ax.fill_between(diff_df.index, mean - std, mean + std,
                    color=COLORS['impact'], alpha=0.2)
    ax.plot(diff_df.index, mean, color=COLORS['impact'], lw=2.5, label='Mean ± std')

    ax.axhline(0, color='black', ls=':', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Impact (Ask - Bid)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)


def generate_all_plots():
    """Generate and save all bid-ask analysis plots."""

    print("Loading bid-ask simulation data...")
    data = load_bidask_data()

    os.makedirs('images', exist_ok=True)

    print("\nGenerating plots...")

    # =========================================================================
    # Queue plots (with scenario: q is reference, bar_q is counterfactual)
    # =========================================================================
    plot_dual_queue_shades(
        data['ask_queue_with'],
        data['bid_queue_with'],
        ask_sim_col='bar_q_a',
        bid_sim_col='bar_q_b',
        title='Bid-Ask Queues: Counterfactual given base queue q',
        ask_ref_col='q_a',
        bid_ref_col='q_b',
        save_path='images/bidask_queue_given_q.png'
    )

    # Queue plots (without scenario: bar_q is reference, q is counterfactual)
    plot_dual_queue_shades(
        data['ask_queue_without'],
        data['bid_queue_without'],
        ask_sim_col='q_a_sim',
        bid_sim_col='q_b_sim',
        title='Bid-Ask Queues: Counterfactual given impacted queue q̄',
        ask_ref_col='bar_q_a',
        bid_ref_col='bar_q_b',
        save_path='images/bidask_queue_given_qbar.png'
    )

    # =========================================================================
    # Merged impact plots (ask - bid on common grid)
    # =========================================================================
    plot_impact_difference(
        data['ask_impact_with'],
        data['bid_impact_with'],
        title='Total Price Impact (Ask - Bid) given base queue q',
        save_path='images/bidask_impact_given_q.png'
    )

    plot_impact_difference(
        data['ask_impact_without'],
        data['bid_impact_without'],
        title='Total Price Impact (Ask - Bid) given impacted queue q̄',
        save_path='images/bidask_impact_given_qbar.png'
    )

    # =========================================================================
    # Individual impact plots for more detail
    # =========================================================================
    plot_impact_shades(
        data['ask_impact_with'],
        title='Ask Side Impact I^a(t) given base queue q',
        color='blue',
        save_path='images/ask_impact_given_q.png'
    )

    plot_impact_shades(
        data['bid_impact_with'],
        title='Bid Side Impact I^b(t) given base queue q',
        color='orange',
        save_path='images/bid_impact_given_q.png'
    )

    plot_impact_shades(
        data['ask_impact_without'],
        title='Ask Side Impact I^a(t) given impacted queue q̄',
        color='blue',
        save_path='images/ask_impact_given_qbar.png'
    )

    plot_impact_shades(
        data['bid_impact_without'],
        title='Bid Side Impact I^b(t) given impacted queue q̄',
        color='orange',
        save_path='images/bid_impact_given_qbar.png'
    )

    print("\nAll plots generated!")


if __name__ == '__main__':
    generate_all_plots()
