import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime, timedelta

def plot_car_energy_usage(user_date):
    # Load and preprocess the data
    df = pd.read_csv(r"114.csv", usecols=['localminute', 'car1'])
    df = df.replace(np.nan, 0)

    # Convert 'localminute' to datetime
    df['localminute'] = pd.to_datetime(df['localminute'])

    # Filter the DataFrame for the selected date
    filtered_df = df[df['localminute'].dt.strftime('%Y-%m-%d') == user_date]
    if filtered_df.empty:
        print("No data available for the selected date.")
        return None, None

    # Generate the array of floats
    array = [float(i) for i in range(1, len(filtered_df) + 1)]

    # Trim or resample df['car1'] to match the length of array if necessary
    df_trimmed = filtered_df['car1'].reset_index(drop=True)

    # Identify peaks in the data
    peaks, _ = find_peaks(df_trimmed)

    # Debugging: Print the peaks
    # print("Peaks found at positions:", peaks)
    # print("Peak values:", df_trimmed[peaks])

    # Process the initial segment if it exists
    if len(peaks) > 0:
        first_peak = peaks[0]
        initial_segment_max = df_trimmed[:first_peak].max()
        if initial_segment_max < 3.0:
            df_trimmed[:first_peak] = 0
    
    last_segment = 0
    # Process the segments between peaks
    for i in range(1, len(peaks)):
        segment_start = peaks[i - 1]
        segment_end = peaks[i]
        last_segment = segment_end + 1
        while ((df_trimmed[segment_start] > 0.0 or df_trimmed[segment_start+1] > 0.0)and df_trimmed[segment_start] < 3.0):
            df_trimmed[segment_start] = 0
            segment_start += 1
        while df_trimmed[segment_end] > 0.0 and df_trimmed[segment_end] < 3.0:
            df_trimmed[segment_end] = 0
            segment_end -= 1

        # # Debugging: Print the current segment range
        # print(f"Segment {i}: Start = {segment_start}, End = {segment_end}")

        segment_max = df_trimmed[segment_start:segment_end].max()
        # print(f"Segment {i} max value: {segment_max}")

        if segment_max < 3.0:
            df_trimmed[segment_start:segment_end] = 0

    # Process the segment after the last peak if it exists
    if len(peaks) > 0 and peaks[-1] < len(df_trimmed) - 1:
        final_segment_max = df_trimmed[last_segment:].max()
        if final_segment_max < 3.0:
            df_trimmed[last_segment:] = 0

    return array, df_trimmed

# def plot_daily_data(user_date):
#     array, df_trimmed = plot_car_energy_usage(user_date)
#     if array is not None and df_trimmed is not None:
#         plt.plot(array, df_trimmed, label='Car Energy Usage (kW)')
#         plt.xlabel('Minutes')
#         plt.ylabel('Car energy usage (kW)')
#         plt.title(f'Car Energy Usage on {user_date}')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#     else:
#         print("No data to plot.")

def plot_weekly_data(start_date):
    # Generate a list of dates for the week
    dates = [start_date + timedelta(days=i) for i in range(7)]

    plt.figure(figsize=(10, 6))

    for date in dates:
        user_date = date.strftime('%Y-%m-%d')
        
        # Call the daily plotting function
        array, df_trimmed = plot_car_energy_usage(user_date)

        if array is not None and df_trimmed is not None:
            # Plot the data
            plt.plot(array, df_trimmed, label=f'{user_date}')

    plt.xlabel('Minutes')
    plt.ylabel('Car energy usage (kW)')
    plt.title('Weekly Car Energy Usage')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    #plot_daily_data('2015-07-12')
    start_date = datetime.strptime('2015-07-12', '%Y-%m-%d')
    plot_weekly_data(start_date)