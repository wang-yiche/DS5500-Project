import pandas as pd
from forecaster_tool.data_processing.pre_processing import handle_missing_value
from forecaster_tool.forecasters.forecaster_collection import forecaster_collection
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error


class ForecasterTool:

    def __init__(self):
        self.data = None  # cleaned raw_data
        self.train = None  # training data
        self.test = None  # test data
        self.target_col = None  # name of target column
        self.freq = None  # frequency of timeseries(length of time interval)
        self.forecaster_type = None  # type of forecaster created
        self.forecaster_instance = None  # forecaster_instance object
        self.test_forecast = None  # forecasts created on test dataset
        self.model = None  # model architecture

    def load_data(self, file, unique_id: int, target: str, features: list, freq: str):
        """
        Load a csv file,

        Args:
            file (str): path to csv file
            unique_id (int): id for station + line combination
            target (str): name of target column
            features (list): list of columns that can be used as features
            freq (str): frequency of the data(interval length, for example '4h' means 4 hour interval data)

        Returns:
            Dataframe containing clean formatted data
        """
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
        """
        create train/test dataframes, split by dates or test_ratio

        Args:
            train_start_date (str): for example: '2017-01-01'
            train_end_date (str):
            test_start_date (str):
            test_end_date (str):
            test_ratio (float): test data ratio

        Returns:
            train and test data
        """
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
        """
        create a forecaster instance with user defined type and parameters

        Args:
            forecaster_type (str): type of forecaster to create
            **kwargs (): additional arguments for the forecaster

        Returns:
            a forecaster instance
        """
        self.forecaster_type = forecaster_type
        self.forecaster_instance = forecaster_collection[forecaster_type](**kwargs)
        self.forecaster_instance.target = self.target_col
        return self.forecaster_instance

    def fit(self):
        """
        train the model on training set, create forecasts on test set

        Returns:
            forecasts df on test set, model
        """
        self.forecaster_instance.preprocessing(self.train, self.test)
        self.test_forecast, self.model = self.forecaster_instance.fit()
        return self.test_forecast, self.model

    def predict(self):

        return

    def save_forecaster(self):

        return

    def load_forecaster(self):

        return

    def calculate_performance(self, df, plot=True, horizon_step=2):
        """
        Calculate model performance on the test set

        Args:
            df (dataframe): forecasts dataframe
            plot (bool): create a plot or not
            horizon_step (int): forecast horizon to plot

        Returns:
            model performance summary dataframe
        """
        if plot:
            df[df['Forecast_Step'] == horizon_step].plot(x="Forecast_Horizon", y=["Actual", "Predicted"])

        perf_dict = {'General': {}}
        y_true = df['Actual']
        y_pred = df['Predicted']
        perf_dict['General']['MAE'] = mean_absolute_error(y_true, y_pred)
        perf_dict['General']['RMSE'] = mean_squared_error(y_true, y_pred, squared=False)
        perf_dict['General']['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
        for i in range(1, self.forecaster_instance.forecast_steps + 1):
            horizon_step_df = df[df['Forecast_Step'] == i]
            y_true = horizon_step_df['Actual']
            y_pred = horizon_step_df['Predicted']
            perf_dict['Horizon_step_' + str(i)] = {}
            perf_dict['Horizon_step_' + str(i)]['MAE'] = mean_absolute_error(y_true, y_pred)
            perf_dict['Horizon_step_' + str(i)]['RMSE'] = mean_squared_error(y_true, y_pred, squared=False)
            perf_dict['Horizon_step_' + str(i)]['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)

        return pd.DataFrame.from_dict(perf_dict)
