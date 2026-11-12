"""Leakage-aware folds: filaments persist for days-weeks, so group by year-month; stratify by station."""
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(df, n_splits=5, seed=42):
    df = df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sgkf.split(df, y=df.station, groups=df.ym)):
        df.loc[df.index[va], "fold"] = k
    return df
