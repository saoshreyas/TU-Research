
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
        while ((df_trimmed[segment_start] > 0.0 or df_trimmed[segment_start + 1] > 0.0) and df_trimmed[segment_start] < 3.0):
            df_trimmed[segment_start] = 0
            segment_start += 1
        while (df_trimmed[segment_end] > 0.0 and df_trimmed[segment_end] < 3.0):
            df_trimmed[segment_end] = 0
            segment_end -= 1

        segment_max = df_trimmed[segment_start:segment_end].max()
        if segment_max < 3.0:
            df_trimmed[segment_start:segment_end] = 0

    # Process the segment after the last peak if it exists
    if len(peaks) > 0 and peaks[-1] < len(df_trimmed) - 1:
        final_segment_max = df_trimmed[last_segment:].max()
        if final_segment_max < 3.0:
            df_trimmed[last_segment:] = 0

    return array, df_trimmed

# Dictionary of binary values to corresponding numbers
binary_dict = {
    "00000": 0,
    "00001": 1,
    "00010": 2,
    "00011": 3,
    "00100": 4,
    "00101": 5,
    "00110": 6,
    "00111": 7,
    "01000": 8,
    "01001": 9,
    "01010": 10,
    "01011": 11,
    "01100": 12,
    "01101": 13,
    "01110": 14,
    "01111": 15,
    "10000": 16,
    "10001": 17,
    "10010": 18,
    "10011": 19,
    "10100": 20,
    "10101": 21,
    "10110": 22,
    "10111": 23,
    "11000": 24,
    "11001": 25,
    "11010": 26,
    "11011": 27,
    "11100": 28,
    "11101": 29,
    "11110": 30,
    "11111": 31,
}

def get_peak_binary_value(df_trimmed):
    # Define the time intervals
    intervals = [
        (0, 4 * 60 + 59),     # 00:00 - 04:59
        (5 * 60, 9 * 60 + 59), # 05:00 - 09:59
        (10 * 60, 13 * 60 + 59),# 10:00 - 13:59
        (14 * 60, 18 * 60 + 59),# 14:00 - 18:59
        (19 * 60, 23 * 60 + 59) # 19:00 - 23:59
    ]

    binary_value = ''

    for start, end in intervals:
        segment = df_trimmed[start:end + 1]
        if segment.max() > 0:
            binary_value += '1'
        else:
            binary_value += '0'

    return binary_value

def plot_daily_data(user_date):
    array, df_trimmed = plot_car_energy_usage(user_date)
    if array is not None and df_trimmed is not None:
        binary_value = get_peak_binary_value(df_trimmed)
        result = binary_dict.get(binary_value, None)
        
        if result is not None:
            print(f"The corresponding value for the binary {binary_value} is: {result}")
        else:
            print(f"No corresponding value found for the binary {binary_value}")

        total_kwh = df_trimmed.sum()
        print(f"Total kWh for the graph: {total_kwh:.2f} kWh")

        plt.plot(array, df_trimmed, label='Car Energy Usage (kW)')
        plt.xlabel('Minutes')
        plt.ylabel('Car energy usage (kW)')
        plt.title(f'Car Energy Usage on {user_date}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data to plot.")

# Example usage
if __name__ == "__main__":
    plot_daily_data('2015-07-15')
