from nbeats_keras.model import NBeatsNet as NBeatsKeras
from forecaster_tool.forecasters.mlp import MLPModel
from forecaster_tool.data_processing.pre_processing import make_time_features, make_mlp_input
from sklearn.preprocessing import MinMaxScaler


class NBeatsModel(MLPModel):

    def preprocessing(self):
        """
        preprocess train and test data for model

        """
        self.scaler = MinMaxScaler()
        self.train_scaled = self.train.copy()
        self.test_scaled = self.test.copy()
        self.train_scaled[self.target] = self.scaler.fit_transform(self.train_scaled[self.target].to_numpy().reshape(-1, 1))
        self.test_scaled[self.target] = self.scaler.transform(self.test_scaled[self.target].to_numpy().reshape(-1, 1))
        self.X_train, self.y_train = make_mlp_input(self.train_scaled, self.look_back_steps, self.target, self.forecast_steps)
        self.X_test, self.y_test = make_mlp_input(self.test_scaled, self.look_back_steps, self.target, self.forecast_steps)
        del self.train_scaled, self.test_scaled

    def get_model_architecture(self):

        model = NBeatsKeras(backcast_length=self.look_back_steps, forecast_length=self.forecast_steps)
        model.compile(optimizer='adam', loss='mae')
        print(model.summary())
        return model