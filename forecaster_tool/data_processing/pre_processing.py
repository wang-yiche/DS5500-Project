import pandas as pd


def handle_missing_value(df: pd.DataFrame, method: str, freq: str):
    idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(idx)
    df = df.interpolate(method=method)
    return df
