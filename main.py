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

# Use Agg backend to save the plot to a buffer
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set your local timezone and the server timezone
LOCAL_TIMEZONE = pytz.timezone("Europe/Budapest")  # Example: Budapest
SERVER_TIMEZONE = pytz.timezone("UTC")  # Assuming server is in UTC

def get_local_time():
    # Get the current time in the server's timezone and convert to local timezone
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

# SQLAlchemy database engine
def create_db_engine():
    return create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

engine = create_db_engine()

def ping_db():
    global engine
    while True:
        time.sleep(600)  # Sleep for 10 minutes
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except OperationalError as err:
            print(f"Error pinging MySQL: {err}")
            engine = create_db_engine()

# Start the background thread to ping the database
thread = threading.Thread(target=ping_db)
thread.daemon = True
thread.start()

# Save contraction to database
def save_contraction_to_db(start, end, duration, severity):
    try:
        print(f"Inserting data into DB: start={start}, end={end}, duration={duration}, severity={severity}")  # Debug
        with engine.connect() as connection:
            query = text("INSERT INTO contractions (start_time, end_time, duration, severity) VALUES (:start, :end, :duration, :severity)")
            connection.execute(query, {"start": start, "end": end, "duration": duration, "severity": severity})
            connection.commit()  # Explicit commit
        print("Data inserted successfully")  # Debug
    except Exception as e:
        print(f"Error inserting data: {e}")  # Debug

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
def predict_birth_time_with_plot(data, rolling_window_size=10, threshold=0.1):
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    
    # Compute intervals between the start of consecutive contractions
    data['interval'] = data['start_time'].diff().dt.total_seconds().fillna(0)
    data['time_since_start'] = (data['start_time'] - data['start_time'].iloc[0]).dt.total_seconds()
    
    # Calculate the rolling standard deviation of intervals and durations
    data['rolling_interval_std'] = data['interval'].rolling(window=rolling_window_size).std()
    data['rolling_duration_std'] = data['duration'].rolling(window=rolling_window_size).std()
    
    # Determine the point where both rolling standard deviations approach zero
    zero_std_index = np.where((data['rolling_interval_std'] <= threshold) & (data['rolling_duration_std'] <= threshold))[0]
    
    if len(zero_std_index) > 0:
        birth_time_seconds = data['time_since_start'].iloc[zero_std_index[0]]
        predicted_birth_time = data['start_time'].iloc[0] + pd.Timedelta(seconds=birth_time_seconds)
    else:
        predicted_birth_time = "Egyelőre nincs elég adat."
    
    # Plotting the data
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Scatter plot of duration vs start_time
    scatter = ax.scatter(data['start_time'], data['duration'], c=data['severity'], cmap='plasma', s=100, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Severity')

    # Calculate the upper and lower bounds of standard deviation
    mean_duration = data['duration'].rolling(window=rolling_window_size).mean()
    upper_bound = mean_duration + data['rolling_duration_std']
    lower_bound = mean_duration - data['rolling_duration_std']
    
    # Plot the mean duration with upper and lower bounds
    ax.plot(data['start_time'], mean_duration, label='Mean Duration', color='green')
    ax.plot(data['start_time'], upper_bound, label='Upper Bound (Mean + SD)', linestyle='--', color='red')
    ax.plot(data['start_time'], lower_bound, label='Lower Bound (Mean - SD)', linestyle='--', color='red')
    
    ax.set_xlabel('Időpont')
    ax.set_ylabel('Időtartam (másodperc)')
    ax.legend()
    ax.set_title('Kontrakciók időtartama az idő függvényében')
    
    img_str = plot_to_base64(fig)
    plt.close(fig)
    
    return predicted_birth_time, img_str

@app.route('/')
def index():
    df = fetch_contractions_from_db()
    if not df.empty:
        predicted_birth_time, img_str = predict_birth_time_with_plot(df)
        print("Plot generated successfully")
    else:
        img_str = None
        predicted_birth_time = None
        print("No data to plot")  # Debug: No data
    
    return render_template('index.html', img_str=img_str, contractions=df.to_dict(orient='records'), predicted_birth_time=predicted_birth_time)

@app.route('/start_timer', methods=['POST'])
def start_timer():
    global current_start_time
    current_start_time = get_local_time()  # Use local time
    return jsonify({'status': 'Timer started', 'start_time': current_start_time.strftime("%Y-%m-%d %H:%M:%S")})

@app.route('/end_timer', methods=['POST'])
def end_timer():
    global current_start_time
    if current_start_time is None:
        return jsonify({'status': 'No timer running'})
    
    end_time = get_local_time()  # Use local time
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
            connection.commit()  # Explicit commit
        print("Data reset successfully")  # Debug
    except Exception as e:
        print(f"Error resetting data: {e}")  # Debug
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=False)
