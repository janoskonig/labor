import os
import base64
import datetime
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import time
import threading



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
        engine = create_db_engine()
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
        engine = create_db_engine()
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

contractions = []
current_start_time = None

@app.route('/')
def index():
    df = fetch_contractions_from_db()
    print(df)  # Debug: print the DataFrame to check its content
    fig, ax = plt.subplots(figsize=(15, 6))
    if not df.empty:
        df['start_time'] = pd.to_datetime(df['start_time'])  # Ensure datetime format
        df.set_index('start_time', inplace=True)
        # Create a custom colormap
        cdict = {
            'red':   [(0.0, 0.0, 0.0),  # Blue
                      (0.5, 0.0, 0.0),  # Green
                      (1.0, 1.0, 1.0)], # Red
            'green': [(0.0, 0.0, 0.0),  # Blue
                      (0.5, 1.0, 1.0),  # Green
                      (1.0, 0.0, 0.0)], # Red
            'blue':  [(0.0, 1.0, 1.0),  # Blue
                      (0.5, 0.0, 0.0),  # Green
                      (1.0, 0.0, 0.0)]  # Red
        }
        custom_cmap = mcolors.LinearSegmentedColormap('BlGrRd', cdict)
        scatter = ax.scatter(df.index, df['duration'], c=df['severity'], cmap=custom_cmap, s=100, alpha=0.7)
        ax.set_xlabel('dátum - időpont')
        ax.set_ylabel('Hossza (seconds)')
        ax.set_title('Kontrakciók')
        plt.xticks(rotation=45)
        plt.colorbar(scatter, ax=ax, label='Erősség')
        img_str = plot_to_base64(fig)
        print("Plot generated successfully")
    else:
        img_str = None
        print("No data to plot")  # Debug: No data
    plt.close(fig)
    return render_template('index.html', img_str=img_str, contractions=contractions)

@app.route('/start_timer', methods=['POST'])
def start_timer():
    global current_start_time
    current_start_time = datetime.datetime.now()
    return jsonify({'status': 'Timer started', 'start_time': current_start_time.strftime("%Y-%m-%d %H:%M:%S")})

@app.route('/end_timer', methods=['POST'])
def end_timer():
    global current_start_time
    if current_start_time is None:
        return jsonify({'status': 'No timer running'})
    
    end_time = datetime.datetime.now()
    duration = (end_time - current_start_time).total_seconds()
    severity = request.json.get('severity', 1)
    contractions.append({'start': current_start_time, 'end': end_time, 'duration': duration, 'severity': severity})
    save_contraction_to_db(current_start_time, end_time, duration, severity)
    current_start_time = None
    return jsonify({'status': 'Timer stopped', 'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"), 'duration': duration, 'severity': severity})

@app.route('/reset')
def reset():
    global contractions
    contractions = []
    try:
        engine = create_db_engine()
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM contractions"))
            connection.commit()  # Explicit commit
        print("Data reset successfully")  # Debug
    except Exception as e:
        print(f"Error resetting data: {e}")  # Debug
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)