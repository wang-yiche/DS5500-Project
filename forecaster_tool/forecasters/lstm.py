from keras.models import Sequential
from keras.layers import LSTM, Dense
from forecaster_tool.forecasters.mlp import MLPModel


class LSTMModel(MLPModel):

    def get_model_architecture(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(self.look_back_steps, self.X_train.shape[2])))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(self.forecast_steps))
        model.compile(optimizer='adam', loss='mse')
        print(model.summary())
        return model
