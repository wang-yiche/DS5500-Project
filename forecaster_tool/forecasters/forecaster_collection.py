from forecaster_tool.forecasters.arima import ArimaModel
from forecaster_tool.forecasters.mlp import MLPModel
from forecaster_tool.forecasters.nbeats import NBeatsModel
from forecaster_tool.forecasters.tcn import TCNModel
from forecaster_tool.forecasters.transformer import Transformer
from forecaster_tool.forecasters.lstm import LSTMModel
from forecaster_tool.forecasters.cnn import CNNModel

# collection of available forecasters
forecaster_collection = {'Arima': ArimaModel, 'MLPModel': MLPModel, 'NBeats': NBeatsModel, 'TCNModel': TCNModel,
                         'Transformer': Transformer, 'LSTM': LSTMModel, 'CNN': CNNModel}
