import torch
import torch.nn as nn
import numpy as np
import json


class WeatherLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


def load_model(model_path='weather_model.pth',
               scaler_path='scaler_params.json'):

    model = WeatherLSTM()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
    )
    model.eval()

    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)

    return model, scaler_params


def scale_data(data, scaler_params):
    min_val = scaler_params['min']
    max_val = scaler_params['max']
    return (data - min_val) / (max_val - min_val)


def inverse_scale(data, scaler_params):
    min_val = scaler_params['min']
    max_val = scaler_params['max']
    return data * (max_val - min_val) + min_val


def predict_next_7_days(model, scaler_params, last_30_temps):
    scaled = scale_data(np.array(last_30_temps), scaler_params)
    sequence = scaled.reshape(-1, 1).tolist()
    predictions = []

    for _ in range(7):
        x = torch.FloatTensor(sequence[-30:]).unsqueeze(0)
        with torch.no_grad():
            next_pred = model(x).item()
        predictions.append(next_pred)
        sequence.append([next_pred])

    predictions = np.array(predictions)
    return inverse_scale(predictions, scaler_params).tolist()
