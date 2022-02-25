import pandas as pd
import pmdarima as pm


class ArimaModel:
    def __init__(self, forecast_steps):
        self.train = None  # training data
        self.test = None  # test data
        self.forecast_steps = forecast_steps  # number of time steps to forecast into the future
        self.forecasts = None  # forecasts dataframe
        self.target = None  # target column
        self.model = None  # model architecture

    def preprocessing(self, train_data, test_data):
        """
        preprocess train and test data for model

        Args:
            train_data (df):
            test_data (df):

        """
        self.train = train_data[[self.target]]
        self.test = test_data[[self.target]]

    def fit(self):
        """
        initiate auto-arima model
        start prediction on test set

        Returns:
            forecast dataframe, model architecture
        """
        self.model = pm.auto_arima(self.train[self.target])
        self.forecasts = self.predict(self.test)
        return self.forecasts, self.model

    def predict(self, data):
        """
        predict using trained model on given dataframe

        Args:
            data (df): df to do forecasts on

        Returns:
            forecasts dataframe
        """
        # train_list = self.train[self.target].tolist()
        test_list = data[self.target].tolist()
        output_list = []
        for test_time_step in test_list:
            # train_list.append(test_time_step)
            self.model.update(test_time_step)
            # self.model = pm.auto_arima(train_list)
            output_list.append(self.model.predict(self.forecast_steps))

        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Datetime', self.target: 'Actual'}, inplace=True)
        forecast_dict = {'Datetime': [],
                         'Actual': [],
                         'Forecast_Horizon': [],
                         'Forecast_Step': [],
                         'Predicted': []}
        window_count = 0
        for index, values in data.iterrows():

            if self.forecast_steps + 1 <= len(data):
                window_df = data[index + 1:index + self.forecast_steps + 1]
            else:
                window_df = data[index + 1: len(data)]

            for i in range(1, len(window_df) + 1):
                forecast_dict['Datetime'].append(values['Datetime'])
                forecast_dict['Forecast_Step'].append(i)
                predicted = int(output_list[window_count].tolist()[i - 1])
                if predicted < 0:
                    predicted = 0
                forecast_dict['Predicted'].append(predicted)

            window_count += 1
            for idx, vls in window_df.iterrows():
                forecast_dict['Forecast_Horizon'].append(vls['Datetime'])
                forecast_dict['Actual'].append(vls['Actual'])

        forecasts = pd.DataFrame.from_dict(forecast_dict)
        return forecasts
