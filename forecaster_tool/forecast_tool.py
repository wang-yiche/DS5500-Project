import pandas as pd
from forecaster_tool.data_processing.pre_processing import handle_missing_value
from forecaster_tool.forecasters.forecaster_collection import forecaster_collection


class ForecasterTool:

    def __init__(self):
        self.data = None
        self.train = None
        self.test = None
        self.target_col = None
        self.freq = None
        self.forecaster_type = None
        self.forecaster_instance = None
        self.test_forecast = None
        self.model = None

    def load_data(self, file, unique_id: int, target: str, features: list, freq: str):
        self.target_col = target
        self.freq = freq
        raw_data = pd.read_csv(file)
        raw_data["Datetime"] = pd.to_datetime(raw_data["Datetime"])
        raw_data.set_index('Datetime', inplace=True)
        features.append(target)
        features.append("Unique ID")
        raw_data = raw_data[features]
        raw_data = raw_data[raw_data['Unique ID'] == unique_id]
        raw_data.drop(columns=['Unique ID'], inplace=True)
        raw_data = handle_missing_value(raw_data, method='time', freq=freq)
        self.data = raw_data
        return raw_data

    def train_test_split(self, train_start_date=None, train_end_date=None,
                         test_start_date=None, test_end_date=None, test_ratio=0.3):
        # slice dataframe using user provided dates
        if None not in (train_start_date, train_end_date, test_start_date, test_end_date):
            self.train = self.data[train_start_date: train_end_date]
            self.test = self.data[test_start_date: test_end_date]
        else:
            test_size = int(len(self.data.index)*test_ratio)
            train_start = self.data[:-test_size].index.min()
            train_end = self.data[:-test_size].index.max()
            train_start_date = train_start.strftime('%Y-%m-%d')
            train_end_date = train_end.strftime('%Y-%m-%d')

            test_start = self.data[-test_size:].index.min()
            test_end = self.data[-test_size:].index.max()
            test_start_date = test_start.strftime('%Y-%m-%d')
            # handle the problem where test_ratio cause last day of training data to collide with first day of test data
            if train_end_date == test_start_date:
                temp_date = test_start + pd.offsets.Day(1)
                test_start_date = temp_date.strftime('%Y-%m-%d')
            test_end_date = test_end.strftime('%Y-%m-%d')

            self.train = self.data[train_start_date: train_end_date]
            self.test = self.data[test_start_date: test_end_date]

        return self.train, self.test

    def create_forecaster(self, forecaster_type: str, **kwargs):
        self.forecaster_type = forecaster_type
        self.forecaster_instance = forecaster_collection[forecaster_type](**kwargs)
        self.forecaster_instance.target = self.target_col
        return self.forecaster_instance

    def fit(self):
        self.forecaster_instance.preprocessing(self.train, self.test)
        self.test_forecast, self.model = self.forecaster_instance.fit()
        return self.test_forecast, self.model

    def predict(self):

        return

    def save_forecaster(self):

        return

    def load_forecaster(self):

        return

    def calculate_performance(self, df):

        return














