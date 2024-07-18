from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist

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

    # Apply standard scalar by dividing each data point by the maximum value in the column
    max_value = df_trimmed.max()
    if max_value != 0:
        df_trimmed = df_trimmed / max_value

    # Identify peaks in the data
    peaks, _ = find_peaks(df_trimmed)

    # Process the initial segment if it exists
    if len(peaks) > 0:
        first_peak = peaks[0]
        initial_segment_max = df_trimmed[:first_peak].max()
        if initial_segment_max < 3.0 / max_value:
            df_trimmed[:first_peak] = 0
    
    last_segment = 0
    # Process the segments between peaks
    for i in range(1, len(peaks)):
        segment_start = peaks[i - 1]
        segment_end = peaks[i]
        last_segment = segment_end + 1
        while ((df_trimmed[segment_start] > 0.0 or df_trimmed[segment_start + 1] > 0.0) and df_trimmed[segment_start] < 3.0 / max_value):
            df_trimmed[segment_start] = 0
            segment_start += 1
        while df_trimmed[segment_end] > 0.0 and df_trimmed[segment_end] < 3.0 / max_value:
            df_trimmed[segment_end] = 0
            segment_end -= 1

        segment_max = df_trimmed[segment_start:segment_end].max()
        if segment_max < 3.0 / max_value:
            df_trimmed[segment_start:segment_end] = 0

    # Process the segment after the last peak if it exists
    if len(peaks) > 0 and peaks[-1] < len(df_trimmed) - 1:
        final_segment_max = df_trimmed[last_segment:].max()
        if final_segment_max < 3.0 / max_value:
            df_trimmed[last_segment:] = 0

    return array, df_trimmed

# Dictionary of binary values to corresponding numbers
binary_dict = {
    "00000": 0,
    "00001": 1,
    "00010": 2,
    "00100": 3,
    "01000": 4,
    "10000": 5,
    "00011": 6,
    "00101": 7,
    "01001": 8,
    "10001": 9,
    "00110": 10,
    "01010": 11,
    "10010": 12,
    "01100": 13,
    "10100": 14,
    "11000": 15,
    "00111": 16,
    "01101": 17,
    "11001": 18,
    "10101": 19,
    "01011": 20,
    "10011": 21,
    "11010": 22,
    "10110": 23,
    "01110": 24,
    "11100": 25,
    "01111": 26,
    "11011": 27,
    "11101": 28,
    "11101": 29,
    "11111": 30,
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

def plot_seasonal_data(start_date):
    # Generate a list of dates for the season (3 months)
    dates = [start_date + timedelta(days=i) for i in range(3 * 31)]

    # Prepare table data
    table_data = []

    for date in dates:
        user_date = date.strftime('%Y-%m-%d')
        
        # Call the daily plotting function
        array, df_trimmed = plot_car_energy_usage(user_date)

        if array is not None and df_trimmed is not None:
            # Get the binary peak value and its corresponding number
            binary_value = get_peak_binary_value(df_trimmed)
            result = binary_dict.get(binary_value, None)
            
            # Calculate total kWh for the day
            total_kwh = ((df_trimmed.sum())/60)
            
            # Append data to table
            table_data.append((result, total_kwh, binary_value))

    # # Print the table
    # print(f"{'Binary Value':<12} {'Total kWh':<10}")
    # print("-" * 25)
    # for row in table_data:
    #     binary_value_str = str(row[0]) if row[0] is not None else "None"
    #     print(f"{binary_value_str:<12} {row[1]:<10.2f}")

    # Convert table data to a DataFrame for clustering
    df_table = pd.DataFrame(table_data, columns=['Binary Value', 'Total kWh', 'Binary Label'])
    
    # Keep original values for labeling
    original_values = df_table[['Binary Value', 'Total kWh', 'Binary Label']].copy()

    # Apply Min-Max scaling for clustering
    scaler = MinMaxScaler()
    df_table[['Binary Value', 'Total kWh']] = scaler.fit_transform(df_table[['Binary Value', 'Total kWh']])

    # Perform hierarchical clustering
    Z = linkage(df_table[['Binary Value', 'Total kWh']], method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

    # Choose the number of clusters by visual inspection of the dendrogram
    num_clusters = 13   # Adjust based on dendrogram
    df_table['Cluster'] = fcluster(Z, num_clusters, criterion='maxclust')

    # Create scatter plot using original values for axes labels
    plt.figure(figsize=(12, 8))

    # Generate distinct colors for each cluster dynamically
    colors = plt.cm.get_cmap('tab20', num_clusters)

    scatter = plt.scatter(original_values['Binary Value'], original_values['Total kWh'], c=df_table['Cluster'], cmap=colors, alpha=0.5)
    
    # Create a legend dynamically based on the number of clusters
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10, label=f'Cluster {i+1}') for i in range(num_clusters)]
    plt.legend(handles=handles, title='Cluster')

    # Add vertical dashed lines at specified x-values if they exist
    for x_value in [5.5, 15.5, 24.5, 28.5]:
        plt.axvline(x=x_value, color='red', linestyle='--', linewidth=1)

    # Calculate centroids and add arrows
    # for cluster in range(1, num_clusters + 1):
    #     cluster_data = original_values[df_table['Cluster'] == cluster]
    #     centroid = cluster_data[['Binary Value', 'Total kWh']].mean().values
    #     closest_point_idx = cdist([centroid], cluster_data[['Binary Value', 'Total kWh']]).argmin()
    #     closest_point = cluster_data.iloc[closest_point_idx]
        
    #     plt.annotate('', xy=(closest_point['Binary Value'], closest_point['Total kWh']), xytext=(centroid[0], centroid[1]),
    #                  arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel('Integer Value')
    plt.ylabel('Total kWh')
    plt.title('Winter 2015 Daily EV Charging Pattern Identification with Hierarchial Clustering')

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start_date = datetime.strptime('2014-12-21', '%Y-%m-%d')
    plot_seasonal_data(start_date)
