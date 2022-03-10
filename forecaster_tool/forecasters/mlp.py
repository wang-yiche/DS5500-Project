import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten
from forecaster_tool.data_processing.pre_processing import make_time_features, make_mlp_input
from sklearn.preprocessing import MinMaxScaler


class MLPModel:
    def __init__(self, forecast_steps, look_back_steps):
        self.look_back_steps = look_back_steps
        self.train = None  # training data
        self.test = None  # test data
        self.forecast_steps = forecast_steps  # number of time steps to forecast into the future
        self.forecasts = None  # forecasts dataframe
        self.target = None  # target column
        self.model = None  # model architecture

    def preprocessing(self):
        """
        preprocess train and test data for model

        """
        self.scaler = MinMaxScaler()
        self.train = make_time_features(self.train)
        self.test = make_time_features(self.test)
        self.train_scaled = self.train.copy()
        self.test_scaled = self.test.copy()
        self.train_scaled[self.target] = self.scaler.fit_transform(self.train_scaled[self.target].to_numpy().reshape(-1, 1))
        self.test_scaled[self.target] = self.scaler.transform(self.test_scaled[self.target].to_numpy().reshape(-1, 1))
        self.X_train, self.y_train = make_mlp_input(self.train_scaled, self.look_back_steps, self.target, self.forecast_steps)
        self.X_test, self.y_test = make_mlp_input(self.test_scaled, self.look_back_steps, self.target, self.forecast_steps)
        del self.train_scaled, self.test_scaled

    def get_model_architecture(self):

        model = Sequential()
        model.add(Dense(50, activation='relu', input_shape=(self.look_back_steps, self.X_train.shape[2])))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.forecast_steps))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def fit(self):
        """
        initiate mlp model
        start prediction on test set

        Returns:
            forecast dataframe, model architecture
        """
        self.model = self.get_model_architecture()
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)
        self.model.fit(self.X_train, self.y_train, epochs=100, verbose=0, validation_split=0.1, callbacks=[es])
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
        output_list = []
        output = self.model.predict(self.X_test)
        for arr in output:
            out = self.scaler.inverse_transform(arr.reshape(-1, 1))
            out = out.reshape(self.forecast_steps)
            output_list.append(out.tolist())

        data.reset_index(inplace=True)
        data = data[self.look_back_steps-1:len(data)-self.forecast_steps].copy()
        data.rename(columns={'index': 'Datetime', self.target: 'Actual'}, inplace=True)
        forecast_dict = {'Datetime': [],
                         'Actual': [],
                         'Forecast_Horizon': [],
                         'Forecast_Step': [],
                         'Predicted': []}
        data.reset_index(inplace=True, drop=True)

        window_count = 0
        for index, values in data.iterrows():

            if self.forecast_steps + index <= len(data):
                window_df = data[index + 1: index + 1 + self.forecast_steps]
            else:
                window_df = data[index+1: len(data)]

            for i in range(1, len(window_df) + 1):
                forecast_dict['Datetime'].append(values['Datetime'])
                forecast_dict['Forecast_Step'].append(i)
                predicted = int(output_list[window_count][i - 1])
                if predicted < 0:
                    predicted = 0
                forecast_dict['Predicted'].append(predicted)

            window_count += 1
            for idx, vls in window_df.iterrows():
                forecast_dict['Forecast_Horizon'].append(vls['Datetime'])

                forecast_dict['Actual'].append(vls['Actual'])

        forecasts = pd.DataFrame.from_dict(forecast_dict)
        return forecasts
