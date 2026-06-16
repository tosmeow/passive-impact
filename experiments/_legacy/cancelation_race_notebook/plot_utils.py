import matplotlib.pyplot as plt
import numpy as np
import os

# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE = os.path.join(SCRIPT_DIR, '../../../data/experiments/cancelation_race')


def load_data():
    """Load cancelation race experiment results."""
    times = np.load(f'{DATA_BASE}/times.npy')
    sample_times = np.load(f'{DATA_BASE}/sample_times.npy')
    queue_paths_grid = np.load(f'{DATA_BASE}/queue_paths_grid.npy')  # [n_sample_times x (1 + n_sims)]
    x_values = np.load(f'{DATA_BASE}/x_values.npy')
    conditioning_cancel_times = np.load(f'{DATA_BASE}/conditioning_cancel_times.npy')
    initial_delta = np.load(f'{DATA_BASE}/initial_delta.npy')[0]

    # Sample grid data - all q̄ paths for different x values
    q_ref_grid = queue_paths_grid[:, 0]
    bar_q_all_grid = queue_paths_grid[:, 1:]  # [n_sample_times x n_x_values]

    return {
        'times': times,
        'sample_times': sample_times,
        'q_ref_grid': q_ref_grid,
        'bar_q_all_grid': bar_q_all_grid,
        'x_values': x_values,
        'conditioning_cancel_times': conditioning_cancel_times,
        'initial_delta': initial_delta,
    }


def plot_queue_trajectories(data, save_path=None):
    """
    Plot all queue trajectories q̄(t; x) for different burst times x.
    Excludes endpoint x values (outliers).
    """
    sample_times = data['sample_times']
    bar_q_all = data['bar_q_all_grid']
    x_values = data['x_values']

    # Exclude endpoints and restrict to burst window
    x_values_inner = x_values[1:-1]
    bar_q_inner = bar_q_all[:, 1:-1]
    t_max = x_values.max()

    fig, ax = plt.subplots(figsize=(12, 6))

    n_sims = bar_q_inner.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_sims))

    for idx in range(n_sims):
        ax.plot(sample_times, bar_q_inner[:, idx], color=colors[idx], linewidth=0.8, alpha=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Queue size $\\bar{q}(t; x)$')
    ax.set_title('Queue Trajectories for Different Burst Times')
    ax.set_xlim(0, t_max)
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=x_values_inner.min(), vmax=x_values_inner.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Burst time x')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_queue_at_fixed_time(data, eval_time=None, save_path=None):
    """
    Plot q̄(t=eval_time; x) as a function of burst time x.
    Shows: where does the queue end up if you burst at different times?
    Excludes the last x value (outlier - burst at end of window).
    """
    sample_times = data['sample_times']
    bar_q_all = data['bar_q_all_grid']
    x_values = data['x_values']

    # Exclude endpoint x values (outliers)
    x_values_plot = x_values[1:-1]
    bar_q_plot = bar_q_all[:, 1:-1]

    # Default to end of burst window
    if eval_time is None:
        eval_time = x_values.max()

    # Find index closest to eval_time
    idx_t = np.argmin(np.abs(sample_times - eval_time))
    actual_t = sample_times[idx_t]

    # Queue at this time for each x (excluding last)
    queue_at_t = bar_q_plot[idx_t, :]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_values_plot, queue_at_t, 'b-', linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Burst time x')
    ax.set_ylabel(f'Queue size $\\bar{{q}}(t={actual_t:.2f}; x)$')
    ax.set_title(f'Queue at t={actual_t:.2f} vs Burst Time')
    ax.grid(True, alpha=0.3)

    # Add annotation for min/max
    min_idx = np.argmin(queue_at_t)
    max_idx = np.argmax(queue_at_t)
    ax.annotate(f'Min: x={x_values_plot[min_idx]:.2f}',
                xy=(x_values_plot[min_idx], queue_at_t[min_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='green')
    ax.annotate(f'Max: x={x_values_plot[max_idx]:.2f}',
                xy=(x_values_plot[max_idx], queue_at_t[max_idx]),
                xytext=(10, -15), textcoords='offset points',
                fontsize=9, color='red')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_cumulative_queue_exposure(data, save_path=None):
    """
    Plot cumulative queue exposure ∫q̄(s; x) ds as a function of burst time x.
    Lower exposure = better (less time exposed to adverse selection).
    """
    sample_times = data['sample_times']
    bar_q_all = data['bar_q_all_grid']
    x_values = data['x_values']

    # Restrict to burst window for integration
    t_max = x_values.max()
    t_mask = sample_times <= t_max
    sample_times_restricted = sample_times[t_mask]
    bar_q_restricted = bar_q_all[t_mask, :]

    # Compute cumulative exposure (trapezoidal integration)
    dt = np.diff(sample_times_restricted)
    n_sims = bar_q_restricted.shape[1]

    cumulative_exposure = np.zeros(n_sims)
    for i in range(n_sims):
        # Trapezoidal rule
        cumulative_exposure[i] = np.sum(0.5 * (bar_q_restricted[:-1, i] + bar_q_restricted[1:, i]) * dt)

    # Exclude endpoints
    x_values_inner = x_values[1:-1]
    cumulative_exposure_inner = cumulative_exposure[1:-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_values_inner, cumulative_exposure_inner, 'g-', linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Burst time x')
    ax.set_ylabel('Cumulative queue exposure $\\int \\bar{q}(s; x) ds$')
    ax.set_title('Total Queue Exposure vs Burst Time')
    ax.grid(True, alpha=0.3)

    # Highlight advantage of early cancel
    min_idx = np.argmin(cumulative_exposure_inner)
    ax.axvline(x_values_inner[min_idx], color='green', linestyle='--', alpha=0.5,
               label=f'Optimal: x={x_values_inner[min_idx]:.2f}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_pairwise_comparison(data, save_path=None):
    """
    Compare early vs late cancelation directly.
    Plot q̄(t; x_early) - q̄(t; x_late) over time for various pairs.
    """
    sample_times = data['sample_times']
    bar_q_all = data['bar_q_all_grid']
    x_values = data['x_values']

    # Exclude endpoint x values (outliers)
    x_values_inner = x_values[1:-1]
    bar_q_inner = bar_q_all[:, 1:-1]

    # Restrict to time before latest burst fires (to see clean comparison period)
    t_max_plot = x_values_inner.max()
    t_mask = sample_times <= t_max_plot

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Compare early vs late (using inner x values only)
    ax = axes[0]
    earliest_idx = 0  # First of inner values
    latest_idx = len(x_values_inner) - 1  # Last of inner values

    diff = bar_q_inner[:, earliest_idx].astype(float) - bar_q_inner[:, latest_idx].astype(float)

    ax.plot(sample_times[t_mask], diff[t_mask], 'b-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x_values_inner[earliest_idx], color='green', linestyle=':', alpha=0.7, label=f'Early burst (x={x_values_inner[earliest_idx]:.2f})')
    ax.axvline(x_values_inner[latest_idx], color='red', linestyle=':', alpha=0.7, label=f'Late burst (x={x_values_inner[latest_idx]:.2f})')

    ax.set_xlabel('Time')
    ax.set_ylabel('$\\bar{q}(t; x_{early}) - \\bar{q}(t; x_{late})$')
    ax.set_title('Earliest vs Latest Burst: Queue Difference')
    ax.set_xlim(0, t_max_plot)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade regions
    ax.fill_between(sample_times[t_mask], 0, diff[t_mask], where=diff[t_mask] < 0, alpha=0.3, color='green', label='Early better')
    ax.fill_between(sample_times[t_mask], 0, diff[t_mask], where=diff[t_mask] > 0, alpha=0.3, color='red', label='Late better')

    # Right: Heatmap of q̄(t; x) - restricted to burst window, exclude endpoint x values
    ax = axes[1]
    bar_q_restricted = bar_q_inner[t_mask, :]
    im = ax.imshow(bar_q_restricted.T, aspect='auto', origin='lower',
                   extent=[0, t_max_plot, x_values_inner.min(), x_values_inner.max()],
                   cmap='viridis', interpolation='bilinear')

    # Draw diagonal line where t = x (burst happens)
    diag_min = max(0, x_values_inner.min())
    diag_max = min(t_max_plot, x_values_inner.max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max],
            'r--', linewidth=2, label='t = x (burst time)')

    ax.set_xlabel('Time t')
    ax.set_ylabel('Burst time x')
    ax.set_title('Queue Heatmap: $\\bar{q}(t; x)$')
    ax.set_xlim(0, t_max_plot)
    ax.set_ylim(x_values_inner.min(), x_values_inner.max())
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Queue size')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_advantage_over_time(data, save_path=None):
    """
    For each time t, show how much better off you are by canceling at different x.
    Plot q̄(t; x) - q̄(t; x_ref) for all x, showing the "advantage" of early cancel.
    Excludes endpoint x values (outliers).
    """
    sample_times = data['sample_times']
    bar_q_all = data['bar_q_all_grid']
    x_values = data['x_values']

    # Exclude endpoints
    x_values_inner = x_values[1:-1]
    bar_q_inner = bar_q_all[:, 1:-1]

    # Restrict to burst window
    t_max = x_values.max()
    t_mask = sample_times <= t_max

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Advantage relative to latest (non-outlier) burst
    ax = axes[0]
    ref_idx = len(x_values_inner) - 1  # Last of inner values
    ref_queue = bar_q_inner[:, ref_idx].astype(float)

    n_sims = bar_q_inner.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_sims))

    for idx in range(n_sims - 1):
        advantage = ref_queue - bar_q_inner[:, idx].astype(float)  # Positive = early is better (lower queue)
        ax.plot(sample_times[t_mask], advantage[t_mask], color=colors[idx], linewidth=0.8, alpha=0.8)

    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('$\\bar{q}(t; x_{ref}) - \\bar{q}(t; x)$')
    ax.set_title(f'Queue Advantage Over Reference (x={x_values_inner[ref_idx]:.1f})')
    ax.set_xlim(0, t_max)
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=x_values_inner.min(), vmax=x_values_inner.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Burst time x')

    # Right: At t=t_max (end of burst window), show advantage as function of x
    ax = axes[1]
    idx_t_end = np.argmin(np.abs(sample_times - t_max))

    advantage_at_end = ref_queue[idx_t_end] - bar_q_inner[idx_t_end, :].astype(float)

    ax.bar(x_values_inner, advantage_at_end, width=x_values_inner[1]-x_values_inner[0], alpha=0.7, color='steelblue')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)

    ax.set_xlabel('Burst time x')
    ax.set_ylabel(f'Queue advantage at t={t_max:.0f}')
    ax.set_title(f'Advantage of Burst at x vs Reference (at t={t_max:.0f})')
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
    data = load_data()

    images_dir = os.path.join(SCRIPT_DIR, 'images')
    os.makedirs(images_dir, exist_ok=True)

    print("Generating cancelation race analysis plots...")

    plot_queue_trajectories(data, save_path=os.path.join(images_dir, 'queue_trajectories.pdf'))
    plot_queue_at_fixed_time(data, eval_time=None, save_path=os.path.join(images_dir, 'queue_at_end.pdf'))
    plot_cumulative_queue_exposure(data, save_path=os.path.join(images_dir, 'cumulative_exposure.pdf'))
    plot_pairwise_comparison(data, save_path=os.path.join(images_dir, 'pairwise_comparison.pdf'))
    plot_advantage_over_time(data, save_path=os.path.join(images_dir, 'advantage_over_time.pdf'))

    print("Done!")


if __name__ == '__main__':
    generate_all_plots()
