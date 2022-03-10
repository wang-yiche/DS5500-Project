from forecaster_tool.forecasters.arima import ArimaModel
from forecaster_tool.forecasters.mlp import MLPModel

# collection of available forecasters
forecaster_collection = {'Arima': ArimaModel, 'MLPModel': MLPModel}
