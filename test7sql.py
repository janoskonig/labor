import os
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, redirect
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from scipy.optimize import curve_fit

load_dotenv(".env")

app = Flask(__name__)
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT", 3306)
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")

# SQLAlchemy database engine
def create_db_engine():
    return create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

engine = create_db_engine()

# Fetch contractions from database
def fetch_contractions_from_db():
    try:
        df = pd.read_sql('SELECT * FROM contractions', con=engine)
        print("Data fetched successfully")  # Debug
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")  # Debug
        return pd.DataFrame()

# Plotting function
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

# Prediction and plotting function
def predict_birth_time_with_plot(data, window_size=5, exclude_fraction=1/4):
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    
    # Calculate the duration of pauses between contractions
    data['pause_duration'] = data['start_time'].shift(-1) - data['end_time']

    # Calculate rolling standard deviation
    rolling_std = data['duration'].rolling(window=window_size).std()
    
    # Fill NaNs with appropriate values
    rolling_std_filled = rolling_std.ffill()

    # Prepare time series for rolling std
    std_time = data['start_time']
    
    # Convert datetime to numerical values for curve fitting
    std_time_numeric = (std_time - std_time.min()).dt.total_seconds()

    # Exclude the first fraction of the rolling standard deviation data
    start_index = int(len(rolling_std_filled) * exclude_fraction)
    filtered_std_time_numeric = std_time_numeric[start_index:]
    filtered_rolling_std = rolling_std_filled[start_index:]

    # Exponential decay function
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the exponential decay function
    popt, pcov = curve_fit(exp_decay, filtered_std_time_numeric, filtered_rolling_std, p0=(1, 0.001, 0), maxfev=10000)

    # Predict rolling standard deviation
    rolling_std_pred = exp_decay(std_time_numeric, *popt)

    # Extend future predictions
    future_times_numeric = np.linspace(std_time_numeric.max() + 1, std_time_numeric.max() + 500, num=500)
    future_rolling_std_pred = exp_decay(future_times_numeric, *popt)

    # Convert numerical future times back to datetime
    future_times = std_time.min() + pd.to_timedelta(future_times_numeric, unit='s')

    # Find the timepoint where rolling_std becomes 0
    childbirth_timepoint = future_times[np.argmin(np.abs(future_rolling_std_pred))]

    # Plot rolling standard deviation with exponential decay fit
    plt.figure(figsize=(14, 7))
    plt.plot(std_time, rolling_std_filled, label='Actual Rolling Std', color='blue')
    plt.plot(std_time, rolling_std_pred, color='green', linewidth=2, label='Exponential Decay Fit')
    plt.plot(future_times, future_rolling_std_pred, label='Future Predictions', color='purple')
    plt.axvline(x=childbirth_timepoint, color='red', linestyle='--', label='Predicted Childbirth')
    plt.xlabel('Start Time')
    plt.ylabel('Rolling Standard Deviation')
    plt.title('Rolling Standard Deviation Over Time with Exponential Decay Fit and Future Predictions')
    plt.legend()
    plt.show()

    # Calculate rolling mean and 2 SD brackets for existing data
    rolling_mean = data['duration'].rolling(window=window_size).mean()
    rolling_std = data['duration'].rolling(window=window_size).std()
    rolling_upper_bound = rolling_mean + (2 * rolling_std)
    rolling_lower_bound = rolling_mean - (2 * rolling_std)

    # Fill NaNs with appropriate values
    rolling_lower_bound_filled = rolling_lower_bound.fillna(method='bfill')
    rolling_upper_bound_filled = rolling_upper_bound.fillna(method='bfill')

    # Convert rolling bounds to numpy arrays
    rolling_lower_bound_array = np.array(rolling_lower_bound_filled, dtype=float)
    rolling_upper_bound_array = np.array(rolling_upper_bound_filled, dtype=float)

    # Plot contraction durations with rolling statistics and predicted childbirth
    plt.figure(figsize=(12, 6))
    plt.scatter(data['start_time'], data['duration'], label='Contraction Durations')
    plt.plot(data['start_time'], rolling_mean, color='r', linestyle='-', label='Rolling Mean Duration')
    plt.plot(data['start_time'], rolling_lower_bound_array, color='g', linestyle='--', label='Rolling Mean ± 2 SD')
    plt.plot(data['start_time'], rolling_upper_bound_array, color='g', linestyle='--')
    plt.fill_between(data['start_time'], rolling_lower_bound_array, rolling_upper_bound_array, color='g', alpha=0.1)
    plt.axvline(x=childbirth_timepoint, color='red', linestyle='--', label='Predicted Childbirth')
    plt.xlabel('Start Time')
    plt.ylabel('Duration (seconds)')
    plt.title('Contraction Durations with Rolling Mean and Rolling 2 SD Brackets')
    plt.legend()
    plt.show()
    
    return childbirth_timepoint

data = fetch_contractions_from_db()
predict_birth_time_with_plot(data, window_size=5, exclude_fraction=1/4)
