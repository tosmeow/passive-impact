import matplotlib.pyplot as plt
import numpy as np
import os

DATA_BASE = '../../../data/experiments/extreme_events'


def load_data():
    """Load extreme events experiment results."""
    # Market order times (for impact)
    times = np.load(f'{DATA_BASE}/times.npy')
    impact_paths = np.load(f'{DATA_BASE}/impact_paths.npy')  # [n_market_times x n_sims]
    queue_paths = np.load(f'{DATA_BASE}/queue_paths.npy')    # [n_market_times x (1 + n_sims)]

    # Sample grid times (for queue diff evolution)
    sample_times = np.load(f'{DATA_BASE}/sample_times.npy')
    queue_paths_grid = np.load(f'{DATA_BASE}/queue_paths_grid.npy')  # [n_sample_times x (1 + n_sims)]
    queue_paths_grid_baseline = np.load(f'{DATA_BASE}/queue_paths_grid_baseline.npy')  # [n_sample_times x (1 + n_sims)]

    initial_deltas = np.load(f'{DATA_BASE}/initial_deltas.npy')
    lc_event_times = np.load(f'{DATA_BASE}/lc_event_times.npy')

    # Market order data
    q_ref = queue_paths[:, 0]
    bar_q_all = queue_paths[:, 1:]

    # Sample grid data (for queue diff plots)
    q_ref_grid = queue_paths_grid[:, 0]
    bar_q_all_grid = queue_paths_grid[:, 1:]
    queue_diff_grid = bar_q_all_grid.astype(float) - q_ref_grid[:, np.newaxis].astype(float)

    # Baseline data (no L/C events)
    q_ref_grid_baseline = queue_paths_grid_baseline[:, 0]
    bar_q_all_grid_baseline = queue_paths_grid_baseline[:, 1:]
    queue_diff_grid_baseline = bar_q_all_grid_baseline.astype(float) - q_ref_grid_baseline[:, np.newaxis].astype(float)

    return {
        'times': times,
        'sample_times': sample_times,
        'impact_paths': impact_paths,
        'q_ref': q_ref,
        'bar_q_all': bar_q_all,
        'q_ref_grid': q_ref_grid,
        'bar_q_all_grid': bar_q_all_grid,
        'queue_diff_grid': queue_diff_grid,
        'queue_diff_grid_baseline': queue_diff_grid_baseline,
        'initial_deltas': initial_deltas,
        'lc_event_times': lc_event_times,
    }


def plot_queue_diff_decay(data, save_path=None):
    """
    Plot 1: Decay of q̄ - q during the L/C event window as a function of initial delta.
    Shows how queue differences decay during extreme events and comparison to baseline.
    Uses sample grid for fine resolution in [0, 1].
    """
    times = data['sample_times']  # Use sample grid
    queue_diff = data['queue_diff_grid']  # Use grid data
    queue_diff_baseline = data['queue_diff_grid_baseline']  # Baseline (no L/C events)
    initial_deltas = data['initial_deltas']
    lc_times = data['lc_event_times']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Time series of q̄ - q for all initial deltas (colorbar instead of legend)
    ax = axes[0]
    n_sims = queue_diff.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_sims))

    # Plot all lines for L/C-conditioned scenario
    for idx in range(n_sims):
        ax.plot(times, queue_diff[:, idx], color=colors[idx], linewidth=0.8, alpha=0.8)

    # Overlay baseline mean normalized difference as dashed red line
    baseline_normalized = queue_diff_baseline / initial_deltas[np.newaxis, :]
    baseline_mean = baseline_normalized.mean(axis=1)
    ax.plot(times, baseline_mean, 'r--', linewidth=2.5, label='Baseline mean (no L/C events)', alpha=0.9)

    # Mark L/C event window
    ax.axvspan(lc_times.min(), lc_times.max(), alpha=0.15, color='orange')

    ax.set_xlabel('Time')
    ax.set_ylabel('q̄ - q')
    ax.set_title('Queue Difference Evolution')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add colorbar for initial delta
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=initial_deltas.min(), vmax=initial_deltas.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Initial Δ₀')

    # Right: q̄ - q at different time snapshots vs initial delta (colorbar for time)
    ax = axes[1]
    # Find time indices for snapshots (within [0, 1])
    snapshot_times_arr = np.linspace(0.0, 1.0, 12)
    colors_snap = plt.cm.plasma(np.linspace(0.1, 0.9, len(snapshot_times_arr)))

    for i, t_snap in enumerate(snapshot_times_arr):
        idx = np.argmin(np.abs(times - t_snap))
        ax.plot(initial_deltas, queue_diff[idx, :], '-', color=colors_snap[i],
                linewidth=1.5, alpha=0.8)

    # Reference line y = x (no decay)
    ax.plot(initial_deltas, initial_deltas, 'k--', alpha=0.5, linewidth=2)

    ax.set_xlabel('Initial Δ₀ = q̄₀ - q₀')
    ax.set_ylabel('q̄(t) - q(t)')
    ax.set_title('Queue Difference vs Initial Perturbation')
    ax.grid(True, alpha=0.3)

    # Add colorbar for time
    sm2 = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax)
    cbar2.set_label('Time t')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_impact_staleness(data, save_path=None):
    """
    Plot 2: Impact evolution during extreme events.
    Shows all paths with colorbar for initial delta.
    """
    times = data['times']
    impact_paths = data['impact_paths']
    initial_deltas = data['initial_deltas']
    lc_times = data['lc_event_times']

    fig, ax = plt.subplots(figsize=(10, 6))

    n_sims = impact_paths.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_sims))

    # Plot all paths
    for idx in range(n_sims):
        ax.plot(times, impact_paths[:, idx], color=colors[idx], linewidth=0.8, alpha=0.8)

    ax.axvspan(lc_times.min(), lc_times.max(), alpha=0.15, color='orange')

    ax.set_xlabel('Time')
    ax.set_ylabel('Impact I(t)')
    ax.set_title('Impact Evolution')
    ax.grid(True, alpha=0.3)

    # Add colorbar for initial delta
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=initial_deltas.min(), vmax=initial_deltas.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Initial Δ₀')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_decay_rate_analysis(data, save_path=None):
    """
    Plot 3: Analyze decay rate of q̄ - q.
    Shows the ratio (q̄(t) - q(t)) / Δ₀ over time and as heatmap, comparing L/C events to baseline.
    Uses sample grid for fine resolution in [0, 1].
    """
    times = data['sample_times']  # Use sample grid
    queue_diff = data['queue_diff_grid']  # Use grid data
    queue_diff_baseline = data['queue_diff_grid_baseline']  # Baseline (no L/C events)
    initial_deltas = data['initial_deltas']
    lc_times = data['lc_event_times']

    # Compute normalized difference: (q̄ - q) / Δ₀
    normalized_diff = queue_diff / initial_deltas[np.newaxis, :]
    normalized_diff_baseline = queue_diff_baseline / initial_deltas[np.newaxis, :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Heatmap of normalized difference (with L/C events)
    ax = axes[0]
    im = ax.imshow(normalized_diff.T, aspect='auto', origin='lower',
                   extent=[times.min(), times.max(), initial_deltas.min(), initial_deltas.max()],
                   cmap='RdYlBu_r', vmin=0, vmax=1.5, interpolation='bilinear')
    ax.axvline(lc_times.max(), color='orange', linestyle='--', linewidth=2, label='End of L/C events')
    ax.set_xlabel('Time')
    ax.set_ylabel('Initial Δ₀')
    ax.set_title('(q̄ - q) / Δ₀  (normalized decay, with L/C events)')
    plt.colorbar(im, ax=ax, label='Normalized difference')
    ax.legend()

    # Right: Mean normalized difference over initial deltas (comparing both scenarios)
    ax = axes[1]
    mean_normalized = normalized_diff.mean(axis=1)
    std_normalized = normalized_diff.std(axis=1)
    mean_normalized_baseline = normalized_diff_baseline.mean(axis=1)
    std_normalized_baseline = normalized_diff_baseline.std(axis=1)

    # Plot L/C-conditioned scenario (blue)
    ax.fill_between(times, mean_normalized - std_normalized, mean_normalized + std_normalized,
                    alpha=0.3, color='blue')
    ax.plot(times, mean_normalized, 'b-', linewidth=2, label='With L/C events ± Std')

    # Plot baseline scenario (red dashed)
    ax.fill_between(times, mean_normalized_baseline - std_normalized_baseline, mean_normalized_baseline + std_normalized_baseline,
                    alpha=0.25, color='red')
    ax.plot(times, mean_normalized_baseline, 'r--', linewidth=2.5, label='Baseline (no L/C events) ± Std')

    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='No decay')
    ax.axvspan(lc_times.min(), lc_times.max(), alpha=0.2, color='orange', label='L/C events')

    ax.set_xlabel('Time')
    ax.set_ylabel('(q̄ - q) / Δ₀')
    ax.set_title('Mean Normalized Queue Difference: Convergence Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_impact_per_unit(data, save_path=None):
    """
    Plot 4: Mean normalized impact I(t) / Δ₀.
    Shows how impact per unit perturbation evolves over time.
    """
    times = data['times']
    impact_paths = data['impact_paths']
    initial_deltas = data['initial_deltas']
    lc_times = data['lc_event_times']

    # Compute normalized impact: I(t) / Δ₀
    normalized_impact = impact_paths / initial_deltas[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(10, 6))

    mean_normalized = normalized_impact.mean(axis=1)
    std_normalized = normalized_impact.std(axis=1)

    ax.fill_between(times, mean_normalized - std_normalized, mean_normalized + std_normalized,
                    alpha=0.3, color='green')
    ax.plot(times, mean_normalized, 'g-', linewidth=2, label='Mean ± Std')
    ax.axvspan(lc_times.min(), lc_times.max(), alpha=0.15, color='orange')
    ax.axvline(lc_times.max(), color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='End of L/C events')

    ax.set_xlabel('Time')
    ax.set_ylabel('I(t) / Δ₀')
    ax.set_title('Mean Normalized Impact (Impact per Unit Perturbation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def generate_all_plots():
    """Generate all experiment analysis plots."""
    import os
    data = load_data()

    # Create output directory if needed
    os.makedirs('images', exist_ok=True)

    print("Generating extreme events analysis plots...")

    plot_queue_diff_decay(data, save_path='images/queue_diff_decay.png')
    plot_impact_staleness(data, save_path='images/impact_staleness.png')
    plot_decay_rate_analysis(data, save_path='images/decay_rate_analysis.png')
    plot_impact_per_unit(data, save_path='images/impact_per_unit.png')

    print("Done!")


if __name__ == '__main__':
    generate_all_plots()
