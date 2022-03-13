from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from forecaster_tool.forecasters.mlp import MLPModel


class CNNModel(MLPModel):

    def get_model_architecture(self):
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(self.look_back_steps, self.X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.forecast_steps))
        model.compile(optimizer='adam', loss='mse')
        print(model.summary())
        return model
