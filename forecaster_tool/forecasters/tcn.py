from forecaster_tool.forecasters.mlp import MLPModel
from keras.models import Sequential
from keras.layers import Dense
from tcn import TCN, tcn_full_summary


class TCNModel(MLPModel):

    def get_model_architecture(self):

        model = Sequential()
        tcn_layer = TCN(input_shape=(self.look_back_steps, self.X_train.shape[2]), kernel_size=2, )
        model.add(tcn_layer)
        model.add(Dense(self.forecast_steps))
        model.compile(optimizer='adam', loss='mae')
        print(tcn_full_summary(model, expand_residual_blocks=False))
        return model
