import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    path_with: pd.DataFrame = pd.read_csv('../impact_paths.csv').set_index('time')
    path_without: pd.DataFrame = pd.read_csv('../impact_paths_without.csv').set_index('time')
    queue_with: pd.DataFrame = pd.read_csv('../queue_paths.csv').set_index('time')
    queue_without: pd.DataFrame = pd.read_csv('../queue_paths_without.csv').set_index('time')
    return path_with, path_without, queue_with, queue_without
    

def plot_queue_shades(df, sim_col, title, label, ref_col = None):
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_cols = [col for col in df.columns if col.startswith(sim_col)]

    for col in sim_cols:
        ax.plot(df.index, df[col], color='gray', alpha=0.1, linewidth=0.5)

    avg_sims = df[sim_cols].mean(axis=1)
    ax.plot(df.index, avg_sims, color='red', linewidth=2.5, label=f'Mean {sim_col}')

    if ref_col is not None:
        ax.plot(df.index, df[ref_col], color='black', linewidth=2.5, label=ref_col)

    ax.set_xlabel(df.index.name)
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()