import os
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from flask import Flask, render_template, request, jsonify, redirect
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import time
import threading
import pytz
from datetime import datetime
from scipy.optimize import curve_fit

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use Agg backend to save the plot to a buffer
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set your local timezone and the server timezone
LOCAL_TIMEZONE = pytz.timezone("Europe/Budapest")
SERVER_TIMEZONE = pytz.timezone("UTC")

def get_local_time():
    server_time = datetime.now(SERVER_TIMEZONE)
    local_time = server_time.astimezone(LOCAL_TIMEZONE)
    return local_time

load_dotenv(".env")

app = Flask(__name__)
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT", 3306)
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")

def create_db_engine():
    return create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

engine = create_db_engine()

def ping_db():
    global engine
    while True:
        time.sleep(600)
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except OperationalError as err:
            logger.error(f"Error pinging MySQL: {err}")
            engine = create_db_engine()

thread = threading.Thread(target=ping_db)
thread.daemon = True
thread.start()

def save_contraction_to_db(start, end, duration, severity):
    try:
        logger.debug(f"Inserting data into DB: start={start}, end={end}, duration={duration}, severity={severity}")
        with engine.connect() as connection:
            query = text("INSERT INTO contractions (start_time, end_time, duration, severity) VALUES (:start, :end, :duration, :severity)")
            connection.execute(query, {"start": start, "end": end, "duration": duration, "severity": severity})
            connection.commit()
        logger.debug("Data inserted successfully")
    except Exception as e:
        logger.error(f"Error inserting data: {e}")

def fetch_contractions_from_db():
    try:
        df = pd.read_sql('SELECT * FROM contractions', con=engine)
        logger.debug("Data fetched successfully")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def predict_birth_time_with_plot(data, window_size=5, exclude_fraction=1/4):
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    
    data['pause_duration'] = data['start_time'].shift(-1) - data['end_time']
    rolling_std = data['duration'].rolling(window=window_size).std()
    rolling_std_filled = rolling_std.ffill()
    std_time = data['start_time']
    std_time_numeric = (std_time - std_time.min()).dt.total_seconds()
    start_index = int(len(rolling_std_filled) * exclude_fraction)
    filtered_std_time_numeric = std_time_numeric[start_index:]
    filtered_rolling_std = rolling_std_filled[start_index:]

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, _ = curve_fit(exp_decay, filtered_std_time_numeric, filtered_rolling_std, p0=(1, 0.001, 0), maxfev=10000)
    rolling_std_pred = exp_decay(std_time_numeric, *popt)
    future_times_numeric = np.linspace(std_time_numeric.max() + 1, std_time_numeric.max() + 500, num=500)
    future_rolling_std_pred = exp_decay(future_times_numeric, *popt)
    future_times = std_time.min() + pd.to_timedelta(future_times_numeric, unit='s')
    predicted_birth_time = future_times[np.argmin(np.abs(future_rolling_std_pred))]

    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(std_time, rolling_std_filled, label='csúszó szórás', color='blue')
    plt.plot(std_time, rolling_std_pred, color='green', linewidth=2, label='exponenciális csökkenés')
    plt.plot(future_times, future_rolling_std_pred, label='Future Predictions', color='purple')
    plt.axvline(x=predicted_birth_time, color='red', linestyle='--', label='Előrejelzett születési idő')
    plt.xlabel('Időpont')
    plt.ylabel('Csúszó szórás')
    plt.title('"Csúszó szórás" (rolling standard deviation) \nexponenciális csökkenése az idő függvényében')
    plt.legend()
    img_str1 = plot_to_base64(fig1)
    plt.close(fig1)

    rolling_mean = data['duration'].rolling(window=window_size).mean()
    rolling_std = data['duration'].rolling(window=window_size).std()
    rolling_upper_bound = rolling_mean + (2 * rolling_std)
    rolling_lower_bound = rolling_mean - (2 * rolling_std)
    rolling_lower_bound_filled = rolling_lower_bound.bfill()
    rolling_upper_bound_filled = rolling_upper_bound.bfill()
    rolling_lower_bound_array = np.array(rolling_lower_bound_filled, dtype=float)
    rolling_upper_bound_array = np.array(rolling_upper_bound_filled, dtype=float)

    fig2 = plt.figure(figsize=(12, 6))
    plt.scatter(data['start_time'], data['duration'], label='Kontrakciók hossza')
    plt.plot(data['start_time'], rolling_mean, color='r', linestyle='-', label=f'Csúszó átlag ({window_size}-es ablak)')
    plt.plot(data['start_time'], rolling_lower_bound_array, color='g', linestyle='--', label='± 2 SD')
    plt.plot(data['start_time'], rolling_upper_bound_array, color='g', linestyle='--')
    plt.fill_between(data['start_time'], rolling_lower_bound_array, rolling_upper_bound_array, color='g', alpha=0.1)
    plt.axvline(x=predicted_birth_time, color='red', linestyle='--', label='Előrejelzett születési idő')
    plt.xlabel('Időpont')
    plt.ylabel('Időtartam (másodperc)')
    plt.title('Kontrakciók hossza csúszó átlaggal és ± 2 szórásos sávokkal')
    plt.legend()
    img_str2 = plot_to_base64(fig2)
    plt.close(fig2)
    
    return predicted_birth_time, img_str1, img_str2
    

@app.route('/')
def index():
    df = fetch_contractions_from_db()
    if not df.empty:
        predicted_birth_time, img_str1, img_str2 = predict_birth_time_with_plot(df)
        logger.info("Plot generated successfully")
    else:
        img_str1 = None
        img_str2 = None
        predicted_birth_time = None
        logger.info("No data to plot")
    
    return render_template('index.html', img_str1=img_str1, img_str2=img_str2, contractions=df.to_dict(orient='records'), predicted_birth_time=predicted_birth_time)

@app.route('/start_timer', methods=['POST'])
def start_timer():
    global current_start_time
    current_start_time = get_local_time()
    return jsonify({'status': 'Timer started', 'start_time': current_start_time.strftime("%Y-%m-%d %H:%M:%S")})

@app.route('/end_timer', methods=['POST'])
def end_timer():
    global current_start_time
    if current_start_time is None:
        return jsonify({'status': 'No timer running'})
    
    end_time = get_local_time()
    duration = (end_time - current_start_time).total_seconds()
    severity = request.json.get('severity', 1)
    save_contraction_to_db(current_start_time, end_time, duration, severity)
    current_start_time = None
    return jsonify({'status': 'Timer stopped', 'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"), 'duration': duration, 'severity': severity})

@app.route('/reset')
def reset():
    try:
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM contractions"))
            connection.commit()
        logger.info("Data reset successfully")
    except Exception as e:
        logger.error(f"Error resetting data: {e}")
    return redirect('/')

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=8000)
