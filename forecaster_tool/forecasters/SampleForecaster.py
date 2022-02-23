import pandas as pd


class SampleForecaster:
    def __init__(self, forecast_steps, arg2):
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.forecast_steps = forecast_steps
        self.forecasts = None

    def preprocessing(self, train_data, test_data):
        self.X_train = train_data
        self.X_test = test_data
        return

    def fit(self):
        self.y_test = self.X_test.copy()
        self.y_test.reset_index(inplace=True)
        self.y_test.rename(columns={'index': 'Datetime', 'Entries': 'Actual'}, inplace=True)
        return self.predict(self.y_test)

    def predict(self, data):
        forecast_dict = {'Datetime': [],
                         'Actual': [],
                         'Forecast_Horizon': [],
                         'Forecast_Step': [],
                         'Predicted': [], }

        for index, values in data.iterrows():
            for i in range(1, self.forecast_steps + 1):
                forecast_dict['Datetime'].append(values['Datetime'])
                forecast_dict['Forecast_Step'].append(i)

            temp_df = data[index + 1:index + self.forecast_steps + 1]
            for idx, vls in temp_df.iterrows():
                forecast_dict['Forecast_Horizon'].append(vls['Datetime'])
                forecast_dict['Predicted'].append(vls['Actual'])
                forecast_dict['Actual'].append(vls['Actual'])

        min_val = min([len(forecast_dict[ele]) for ele in forecast_dict])
        for ele in forecast_dict:
            forecast_dict[ele] = forecast_dict[ele][:min_val]

        self.forecasts = pd.DataFrame.from_dict(forecast_dict)
        return self.forecasts
