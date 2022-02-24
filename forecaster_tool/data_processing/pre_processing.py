import pandas as pd


def handle_missing_value(df: pd.DataFrame, method: str, freq: str):
    """
    Examine missing dates and impute missing values

    Args:
        df (dataframe): timeseries dataframe
        method (str): method used for imputation
        freq (str): time series interval length

    Returns:
        cleaned df
    """
    idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(idx)
    df = df.interpolate(method=method)
    return df
