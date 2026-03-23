from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
from datetime import datetime, timedelta
from model import load_model, predict_next_7_days

app = Flask(__name__)
model, scaler_params = load_model()
print("Model loaded!")


def get_coordinates(city_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1, "language": "en"}
    response = requests.get(url, params=params)
    data = response.json()
    if 'results' not in data or len(data['results']) == 0:
        return None, None, None
    result = data['results'][0]
    return result['latitude'], result['longitude'], result['name']


def get_weather_data(latitude, longitude):
    end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=62)).strftime('%Y-%m-%d')

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,relative_humidity_2m_max",
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'daily' not in data:
        raise Exception(f"API error: {data}")

    daily = data['daily']

    def clean(arr):
        arr = [x for x in arr if x is not None]
        return arr[-30:]

    return {
        'temp_max':  clean(daily['temperature_2m_max']),
        'temp_min':  clean(daily['temperature_2m_min']),
        'rain':      clean(daily['precipitation_sum']),
        'wind':      clean(daily['windspeed_10m_max']),
        'humidity':  clean(daily['relative_humidity_2m_max'])
    }


def get_forecast(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation_probability",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max,relative_humidity_2m_max,weathercode",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.json.get('city')
        if not city:
            return jsonify({'error': 'City name required'}), 400

        lat, lon, city_name = get_coordinates(city)
        if lat is None:
            return jsonify({'error': 'City not found'}), 404

        historical = get_weather_data(lat, lon)
        forecast = get_forecast(lat, lon)

        if 'hourly' not in forecast:
            raise Exception(f"Forecast API error: {forecast}")

        predictions = predict_next_7_days(
            model, scaler_params, historical['temp_max']
        )

        days = []
        for i in range(7):
            day = (datetime.now() + timedelta(days=i+1)).strftime('%d %b')
            days.append(day)

        f = forecast

        return jsonify({
            'city':              city_name,
            'predictions':       [round(p, 1) for p in predictions],
            'days':              days,
            'current_temp':      round(historical['temp_max'][-1], 1),
            'current_min':       round(historical['temp_min'][-1], 1),
            'humidity':          round(historical['humidity'][-1], 1),
            'wind':              round(historical['wind'][-1], 1),
            'rain_chance':       f['daily']['precipitation_probability_max'][:7],
            'forecast_max':      [round(x, 1) for x in f['daily']['temperature_2m_max'][:7]],
            'forecast_min':      [round(x, 1) for x in f['daily']['temperature_2m_min'][:7]],
            'weathercode':       f['daily']['weathercode'][0],
            'hourly_temps':      f['hourly']['temperature_2m'][:24],
            'hourly_times':      f['hourly']['time'][:24],
            'lat': lat,
            'lon': lon
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
