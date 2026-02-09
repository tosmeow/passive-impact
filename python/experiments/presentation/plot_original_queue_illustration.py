"""
Plot Original Markovian Queue Trajectory

Reads NPY data from Rust simulation and creates a 4-panel figure showing
the original trajectory only:
1. Limit order intensity λ^L(q) over time
2. Cancel intensity λ^C(q) over time
3. Hawkes (market order) intensity over time
4. Queue size over time
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    script_dir = Path(__file__).parent

    # Load NPY files
    times = np.load(script_dir / 'times.npy')
    queues = np.load(script_dir / 'queues.npy')  # shape: (n_times, 1 + n_counterfactual)
    params = np.load(script_dir / 'params.npy')  # [a_l, b_l, a_c, b_c]
    hawkes_intensity = np.load(script_dir / 'hawkes_intensity.npy')

    a_l, b_l, a_c, b_c = params

    # Extract original queue only
    q_orig = queues[:, 0]

    # Compute limit and cancel intensities
    lambda_L_orig = np.maximum(a_l + b_l * q_orig, 0)
    lambda_C_orig = np.maximum(a_c + b_c * q_orig, 0)

    print(f"Loaded: {len(times)} time points")

    # Determine time range
    T_max = times[-1]

    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Common styling
    line_color = 'black'
    line_width = 1.5

    # Plot 1: λ^L over time (top-left)
    ax1 = axes[0, 0]
    ax1.plot(times, lambda_L_orig, color=line_color, linewidth=line_width)
    ax1.set_xlabel('Time $t$', fontsize=12)
    ax1.set_ylabel(r'$\lambda^L(q_t)$', fontsize=12)
    ax1.set_title('Limit Order Intensity', fontsize=13)
    ax1.set_xlim(0, T_max)
    ax1.grid(True, alpha=0.3)

    # Plot 2: λ^C over time (top-right)
    ax2 = axes[0, 1]
    ax2.plot(times, lambda_C_orig, color=line_color, linewidth=line_width)
    ax2.set_xlabel('Time $t$', fontsize=12)
    ax2.set_ylabel(r'$\lambda^C(q_t)$', fontsize=12)
    ax2.set_title('Cancel Intensity', fontsize=13)
    ax2.set_xlim(0, T_max)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Hawkes intensity over time (bottom-left)
    ax3 = axes[1, 0]
    ax3.plot(times, hawkes_intensity, color=line_color, linewidth=line_width)
    ax3.set_xlabel('Time $t$', fontsize=12)
    ax3.set_ylabel(r'$\lambda^M(t)$', fontsize=12)
    ax3.set_title('Market Order Intensity (Hawkes)', fontsize=13)
    ax3.set_xlim(0, T_max)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Queue size over time (bottom-right)
    ax4 = axes[1, 1]
    ax4.plot(times, q_orig, color=line_color, linewidth=line_width)
    ax4.set_xlabel('Time $t$', fontsize=12)
    ax4.set_ylabel('Queue size $q_t$', fontsize=12)
    ax4.set_title('Queue Dynamics', fontsize=13)
    ax4.set_xlim(0, T_max)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_png = script_dir / 'original_markovian_queue_illustration.png'
    fig.savefig(output_png, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_png}")

    output_pdf = script_dir / 'original_markovian_queue_illustration.pdf'
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_pdf}")

    plt.show()


if __name__ == '__main__':
    main()
