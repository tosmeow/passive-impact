"""
Plot Markovian Queue Illustration

Reads NPY data from Rust simulation and creates a 3-panel figure showing:
1. Limit order intensity λ^L(q) over time
2. Cancel intensity λ^C(q) over time
3. Queue sizes over time
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path


def main():
    script_dir = Path(__file__).parent

    # Load NPY files
    times = np.load(script_dir / 'times.npy')
    queues = np.load(script_dir / 'queues.npy')  # shape: (n_times, 1 + n_counterfactual)
    params = np.load(script_dir / 'params.npy')  # [a_l, b_l, a_c, b_c]

    a_l, b_l, a_c, b_c = params

    # Extract original and counterfactual queues
    q_orig = queues[:, 0]
    q_cfs = queues[:, 1:]  # shape: (n_times, n_counterfactual)
    n_counterfactual = q_cfs.shape[1]

    # Compute intensities
    lambda_L_orig = np.maximum(a_l + b_l * q_orig, 0)
    lambda_C_orig = np.maximum(a_c + b_c * q_orig, 0)

    lambda_L_cfs = np.maximum(a_l + b_l * q_cfs, 0)
    lambda_C_cfs = np.maximum(a_c + b_c * q_cfs, 0)

    print(f"Loaded: {len(times)} time points, {n_counterfactual} counterfactual paths")

    # Determine time range
    T_max = times[-1]

    # Create figure with 3 side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Colors - use alpha for many paths
    orig_color = 'black'
    cf_color = 'steelblue'
    cf_alpha = min(0.5, 5.0 / n_counterfactual)  # Adjust alpha based on number of paths

    # Plot 1: λ^L over time
    ax1 = axes[0]
    for i in range(n_counterfactual):
        ax1.plot(times, lambda_L_cfs[:, i], color=cf_color, linewidth=0.5, alpha=cf_alpha)
    ax1.plot(times, lambda_L_orig, color=orig_color, linewidth=2)
    ax1.set_xlabel('Time $t$', fontsize=12)
    ax1.set_ylabel(r'$\lambda^L(q_t)$', fontsize=12)
    ax1.set_title('Limit Order Intensity', fontsize=13)
    ax1.set_xlim(0, T_max)
    ax1.grid(True, alpha=0.3)

    # Plot 2: λ^C over time
    ax2 = axes[1]
    for i in range(n_counterfactual):
        ax2.plot(times, lambda_C_cfs[:, i], color=cf_color, linewidth=0.5, alpha=cf_alpha)
    ax2.plot(times, lambda_C_orig, color=orig_color, linewidth=2)
    ax2.set_xlabel('Time $t$', fontsize=12)
    ax2.set_ylabel(r'$\lambda^C(q_t)$', fontsize=12)
    ax2.set_title('Cancel Intensity', fontsize=13)
    ax2.set_xlim(0, T_max)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Queue sizes
    ax3 = axes[2]
    for i in range(n_counterfactual):
        ax3.plot(times, q_cfs[:, i], color=cf_color, linewidth=0.5, alpha=cf_alpha)
    ax3.plot(times, q_orig, color=orig_color, linewidth=2)
    ax3.set_xlabel('Time $t$', fontsize=12)
    ax3.set_ylabel('Queue size', fontsize=12)
    ax3.set_title('Queue Dynamics', fontsize=13)
    ax3.set_xlim(0, T_max)
    ax3.grid(True, alpha=0.3)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color=orig_color, linewidth=2, label='Original $q$'),
        Line2D([0], [0], color=cf_color, linewidth=1.5, alpha=0.7,
               label=r'Counterfactual $\bar{q}$')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save figure
    output_png = script_dir / 'markovian_queue_illustration.png'
    fig.savefig(output_png, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_png}")

    output_pdf = script_dir / 'markovian_queue_illustration.pdf'
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_pdf}")

    plt.show()


if __name__ == '__main__':
    main()
