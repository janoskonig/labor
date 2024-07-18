import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def analyze_contractions(data, window_size=5, exclude_fraction=1/4):
    number_of_contractions = len(data)

    # Calculate rolling standard deviation
    rolling_std = data['duration'].rolling(window=window_size).std()

    # Fill NaNs with appropriate values
    rolling_std_filled = rolling_std.fillna(method='bfill')

    # Prepare time series for rolling std
    std_time = data['start_time']

    # Exclude the first fraction of the rolling standard deviation data
    start_index = int(len(rolling_std_filled) * exclude_fraction)
    filtered_std_time = std_time[start_index:]
    filtered_rolling_std = rolling_std_filled[start_index:]

    # Exponential decay function
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the exponential decay function
    popt, pcov = curve_fit(exp_decay, filtered_std_time, filtered_rolling_std, p0=(1, 0.001, 0), maxfev=10000)

    # Predict rolling standard deviation
    rolling_std_pred = exp_decay(std_time, *popt)

    # Extend future predictions
    future_times = np.linspace(std_time.max() + 1, std_time.max() + 500, num=500)
    future_rolling_std_pred = exp_decay(future_times, *popt)

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

# Example usage
data = pd.read_csv('simulated_contractions.csv')
childbirth_timepoint = analyze_contractions(data, window_size=5, exclude_fraction=1/4)
print("Predicted Childbirth Timepoint:", childbirth_timepoint)
