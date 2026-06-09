import numpy as np
import pandas as pd


def mapping_function(n_orders, n_traders, freq_dist="powerlaw",
                     alpha=1.5, f0=1.0, seed=None):

    rng = np.random.default_rng(seed)
    N = int(n_traders)

    if freq_dist == "powerlaw":
        if alpha <= 1:
            raise ValueError("alpha must be > 1")
        f = 1.0 + rng.pareto(alpha - 1.0, size=N)
    elif freq_dist == "homogeneous":
        f = np.full(N, float(f0))
    else:
        raise ValueError("freq_dist must be 'powerlaw' or 'homogeneous'")

    p = f / f.sum()
    c = np.concatenate(([0.0], np.cumsum(p)))
    c[-1] = 1.0
    U = rng.random(n_orders)
    return np.clip(np.searchsorted(c, U, side="right") - 1, 0, N - 1)


def assign_metaorders(trades, *, time_col="timestamp", sign_col="sign",
                      n_traders=500, freq_dist="powerlaw", alpha=1.5, f0=1.0,
                      min_children=5, seed=None):
    df = trades.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)


    df["trader_id"] = mapping_function(
        len(df), n_traders, freq_dist=freq_dist, alpha=alpha, f0=f0, seed=seed
    )

    df = df.sort_values(["trader_id", time_col]).reset_index(drop=True)
    new_block = (
        (df["trader_id"] != df["trader_id"].shift())
        | (df[sign_col] != df[sign_col].shift())
    )
    df["metaorder_id"] = new_block.cumsum()

    sizes = df.groupby("metaorder_id")["metaorder_id"].transform("size")
    return df[sizes >= min_children].reset_index(drop=True)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 10_000
    demo = pd.DataFrame({
        "timestamp": pd.Timestamp("2024-01-15 09:00") +
                     pd.to_timedelta(np.sort(rng.uniform(0, 28800, n)), unit="s"),
        "price": 100 + np.cumsum(rng.normal(0, 0.01, n)),
        "quantity": rng.integers(1, 500, n),
        "sign": rng.choice([-1, 1], n),
    })

    mo = assign_metaorders(demo, n_traders=300, alpha=1.5, seed=42)
    print(f"{mo['metaorder_id'].nunique()} metaorders over {len(mo)} trades")
    print(mo.head(8).to_string(index=False))
