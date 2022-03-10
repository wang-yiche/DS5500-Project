import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def make_time_features(df: pd.DataFrame):

    def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
        return data

    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df = encode(df, 'hour_of_day', 23)
    df.drop(columns=['hour_of_day'], inplace=True)

    return df


def make_mlp_input(df: pd.DataFrame, look_back_steps, target, forecast_steps):
    row_no = 0
    row_list = []

    for index, values in df.iterrows():
        if row_no + look_back_steps <= len(df[:-forecast_steps]):
            window_df = df[:-forecast_steps][row_no:row_no + look_back_steps]
            row_list.append(np.array(window_df))
            row_no += 1
        else:
            break

    row_no = 0
    output_list = []
    for index, values in df[look_back_steps:].iterrows():
        if row_no + forecast_steps <= len(df[look_back_steps:]):
            window_df = df[look_back_steps:][target][row_no:row_no + forecast_steps]
            output_list.append(np.array(window_df))
            row_no += 1
        else:
            break

    mlp_X = np.reshape(row_list, (len(row_list), look_back_steps, df.shape[1]))
    mlp_y = np.reshape(output_list, (len(output_list), forecast_steps))

    return mlp_X, mlp_y
