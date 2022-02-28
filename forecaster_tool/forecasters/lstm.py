import pandas as pd
import pmdarima as pm
import torch
import torch.nn as nn







class LSTMModel(nn.Module):
    def __init__(self):
        self.train = None  # training data
        self.test = None  # test data
        self.forecast_steps = forecast_steps  # number of time steps to forecast into the future
        self.forecasts = None  # forecasts dataframe
        self.target = None  # target column
        self.model = None  # model architecture

        super().__init__()
        self.target = target # number of variables
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.linear(hidden_size, output_size)
        self.hidden_layer_size = hidden_size
        self.to_hidden = nn.LSTM(inp, hidden_size)
        self.hidden_cell = (
            torch.zeros(target, 1, self.hidden_layer_size),
            torch.zeros(target, 1, self.hidden_layer_size)
        )

    def forward(self, inp):
        inp = self.dropout(inp)
        output, hidden = self.to_hidden(inp.view(len(inp), target, 1), self.hidden_layer_size)
        pred = self.dense(output.view(len(inp), -1))
        return pred[-1]

    def one_hot(self, str_list):
        list_set = list(set(str_list))
        tmp_dic = {}
        for idx, element in enumerate(list_set):
            tmp_dic[element] = [0] * len(list_set)
            tmp_dic[element][idx] = 1
        one_list = []
        for i in str_list:
            one_list.append(tmp_dic[i])
        return one_list

    def preprocessing(self, data_name):
        df = pd.read(data_name)
        for col_name in df.columns:
            for i in df[col_name].tolist():
                if type(i) == str:
                    df[col_name] = one_hot(df[col_name].to_list)
            if col_name == 'Datetime':
                for i in df[col_name].tolist():
                    y, m, d = str(i).split('/')
        return df

    def fit(self, x, y, model, iter, bb):
        bb =  bb # 'how many steps you want to save a model'
        model = LSTMModel()
        while i < iter:
            optimizer.zero_grad()
            preds = model.pred(x)
            ls = loss(y, preds)
            ls.backward()
            i += 1
            if i % bb == 0:
                torch.save()

    def predict(self, data):
        return forecasts;
























