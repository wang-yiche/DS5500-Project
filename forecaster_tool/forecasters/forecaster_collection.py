from forecaster_tool.forecasters.arima import ArimaModel
from forecaster_tool.forecasters.mlp import MLPModel
from forecaster_tool.forecasters.nbeats import NBeatsModel
from forecaster_tool.forecasters.tcn import TCNModel

# collection of available forecasters
forecaster_collection = {'Arima': ArimaModel, 'MLPModel': MLPModel, 'NBeats': NBeatsModel, 'TCNModel': TCNModel}
