import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiments.plot_utils_common import (
    maybe_set_title,
    save_or_show,
    with_output_format,
)

LOAD_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_BASE = LOAD_EXPERIMENT_DIR / 'data'


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
    return LOAD_EXPERIMENT_DIR / 'images'


def _default_data_base(counterfactual):
    scenario = 'without' if counterfactual else 'with'
    scenario_base = DATA_BASE / scenario
    if (
        (scenario_base / 'times.npy').exists()
        and (scenario_base / 'queue_paths.npy').exists()
        and (scenario_base / 'impact_paths.npy').exists()
        and (scenario_base / 'event_types.npy').exists()
    ):
        return scenario_base
    return DATA_BASE


def _conditioning_suffix(counterfactual):
    return 'given_qbar' if counterfactual else 'given_q'


def _default_mean_label(sim_prefix, ylabel):
    if 'impact' in ylabel.lower():
        return 'Mean impact'
    if sim_prefix.startswith('q_sim'):
        return 'Mean q'
    if sim_prefix.startswith('bar_q'):
        return 'Mean $\\bar{q}$'
    return 'Mean'


def _display_col_label(col):
    if col == 'bar_q':
        return '$\\bar{q}$'
    return col


def _finite_values(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _plot_ylim(
    df,
    sim_prefix,
    ref_col=None,
    lower_padding=0.05,
    top_headroom=0.20,
    path_quantile=0.99,
):
    sim_cols = [col for col in df.columns if col.startswith(sim_prefix)]
    if not sim_cols:
        raise ValueError(f"No simulation columns found with prefix {sim_prefix!r}")

    sim_values = _finite_values(df[sim_cols].to_numpy())
    if sim_values.size == 0:
        raise ValueError("No finite simulation values found for y-axis limits")

    mean_values = _finite_values(df[sim_cols].mean(axis=1).to_numpy())
    y_min_candidates = [
        float(np.quantile(sim_values, 1.0 - path_quantile)),
        float(mean_values.min()),
    ]
    y_max_candidates = [
        float(np.quantile(sim_values, path_quantile)),
        float(mean_values.max()),
    ]

    if ref_col is not None and ref_col in df.columns:
        ref_values = _finite_values(df[ref_col].to_numpy())
        if ref_values.size:
            y_min_candidates.append(float(ref_values.min()))
            y_max_candidates.append(float(ref_values.max()))

    ymin = min(y_min_candidates)
    ymax = max(y_max_candidates)
    span = ymax - ymin
    if span == 0:
        span = max(abs(ymax), 1.0)

    lower = ymin - lower_padding * span
    upper = lower + (ymax - lower) / (1.0 - top_headroom)
    return lower, upper


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


def load_data(counterfactual=False, data_base=None, bar_kappa=None):
    """Load aggressive impact simulation results from .npy files."""
    data_base = Path(data_base) if data_base is not None else _default_data_base(counterfactual)
    times = np.load(data_base / 'times.npy')
    impact_paths = np.load(data_base / 'impact_paths.npy')
    queue_paths = np.load(data_base / 'queue_paths.npy')
    event_types = np.load(data_base / 'event_types.npy')
    if bar_kappa is None:
        bar_kappa_path = data_base / 'bar_kappa.npy'
        bar_kappa = float(np.load(bar_kappa_path)[0]) if bar_kappa_path.exists() else None

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

    meta_mask = ~is_market
    meta_end = times[meta_mask].max() if meta_mask.any() else None

    return impact_df, queue_df, is_market, meta_end, bar_kappa


def compute_plot_y_lims(counterfactual=False, data_base=None, bar_kappa=None):
    """Return padded y-limits for the plots generated from one data set."""
    impact_df, queue_df, _is_market, _meta_end, _bar_kappa = load_data(
        counterfactual=counterfactual,
        data_base=data_base,
        bar_kappa=bar_kappa,
    )
    layout = _queue_layout(counterfactual)
    return {
        'impact': _plot_ylim(impact_df, 'sim_'),
        'queue': _plot_ylim(queue_df, layout['sim_prefix'], ref_col=layout['ref_col']),
    }


def plot_shades(
    df,
    sim_prefix,
    title,
    ylabel,
    meta_end=None,
    ref_col=None,
    save_path=None,
    mean_label=None,
    include_title=False,
    y_lim=None,
):
    """Plot individual simulation paths as transparent lines with mean overlay."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in df.columns if col.startswith(sim_prefix)]

    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.05, linewidth=0.5)

    if not sim_cols:
        raise ValueError(f"No simulation columns found with prefix {sim_prefix!r}")

    if mean_label is None:
        mean_label = _default_mean_label(sim_prefix, ylabel)

    avg = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg, color='red', linewidth=2.5, label=mean_label)

    if ref_col is not None and ref_col in df.columns:
        ax.plot(
            df.index,
            df[ref_col],
            color='black',
            linewidth=2.5,
            label=_display_col_label(ref_col),
        )

    if meta_end is not None:
        ax.axvline(x=meta_end, color='green', linestyle='--', label='End of metaorder')

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(ylabel)
    maybe_set_title(ax, title, include_title)
    ax.legend()
    plt.tight_layout()

    save_or_show(fig, save_path, dpi=300)


def plot_impact_decomposition(
    impact_df,
    is_market,
    bar_kappa,
    meta_end=None,
    save_path=None,
    include_title=False,
):
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
    maybe_set_title(ax, 'Aggressive Impact by Event Type', include_title)
    ax.legend()
    plt.tight_layout()

    save_or_show(fig, save_path, dpi=300)


def plot_queue_diff(
    queue_df,
    counterfactual=False,
    meta_end=None,
    save_path=None,
    include_title=False,
):
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
    maybe_set_title(ax, 'Queue depletion from aggressive meta orders', include_title)
    ax.legend()
    plt.tight_layout()

    save_or_show(fig, save_path, dpi=300)


def generate_all_plots(
    counterfactual=False,
    data_base=None,
    output_dir=None,
    bar_kappa=None,
    include_title=False,
    y_lims=None,
    output_format='pdf',
):
    """Generate and save all analysis plots."""
    impact_df, queue_df, is_market, meta_end, bar_kappa = load_data(
        counterfactual=counterfactual,
        data_base=data_base,
        bar_kappa=bar_kappa,
    )
    layout = _queue_layout(counterfactual)

    output_dir = Path(output_dir) if output_dir is not None else _default_output_dir(counterfactual)

    direction = 'without us' if counterfactual else 'with us'
    suffix = _conditioning_suffix(counterfactual)
    meta_msg = f"{meta_end:.2f}" if meta_end is not None else "unknown"
    kappa_msg = f"{bar_kappa:.4f}" if bar_kappa is not None else "n/a"
    print(
        f"Generating {direction} plots "
        f"(bar_kappa={kappa_msg}, metaorder ends at t={meta_msg})..."
    )

    plot_shades(
        impact_df,
        sim_prefix='sim_',
        title=r'Aggressive Market Impact MI(t)',
        ylabel='Price Impact',
        meta_end=meta_end,
        save_path=with_output_format(output_dir / f'impact_paths_{suffix}.pdf', output_format),
        mean_label='Mean impact',
        include_title=include_title,
        y_lim=y_lims.get('impact') if y_lims else None,
    )

    plot_shades(
        queue_df,
        sim_prefix=layout['sim_prefix'],
        title=layout['queue_title'],
        ylabel='Queue Size',
        meta_end=meta_end,
        ref_col=layout['ref_col'],
        save_path=with_output_format(output_dir / f'queue_paths_{suffix}.pdf', output_format),
        mean_label=layout['mean_label'],
        include_title=include_title,
        y_lim=y_lims.get('queue') if y_lims else None,
    )


if __name__ == '__main__':
    generate_all_plots()
