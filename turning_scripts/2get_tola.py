# This script will detect takeoffs and landings
import pandas as pd

import os, sys
# add parent directory to sys.path to import path_prefix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_prefix import PATH_PREFIX
import numpy as np
from scipy import stats
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

def resample_data(t: np.array, h: np.array, lats: np.array, lons: np.array, freq = 1):
    # use linear interpolation to resample the data
    t_resampled = np.arange(t[0], t[-1], freq)
    h_resampled = np.interp(t_resampled, t, h)
    lats_resampled = np.interp(t_resampled, t, lats)
    lons_resampled = np.interp(t_resampled, t, lons)
    return t_resampled, h_resampled, lats_resampled, lons_resampled

def tola_slope_detector(t: np.array, h: np.array, window_size: int = 10, slope_threshold_takeoff: float = 10, slope_threshold_landing: float = -10, transition_alt: float = 1200, transition_alt_threshold = 500, mode = 'to'):
    """
    The logic for detecting takeoff is:
    - find the index of the first point where h is greater than transition_alt
    - define t_window_end is that index 
    - define t_window_start is that index - window_size
    - if t_window_start is less than 0, then set it to 0
    - if t_window_end is greater than the length of t, then set it to the length of t
    - find the slope of the linear regression line between t_window_start and t_window_end
    - if the slope is greater than the slope_threshold, then it is a takeoff
    - if the slope is less than the slope_threshold, then it is a landing
    """

    if mode == 'to':
        # Find the index of the first point where h is greater than transition_alt
        try:
            transition_index = np.where(h > transition_alt)[0][0]
        except IndexError:
            transition_index = np.nan
    elif mode == 'la':
        # Find the index of the first point where h is still less than transition_alt
        try:
            transition_index = np.where(h < transition_alt)[0][0]
        except IndexError:
            transition_index = np.nan
    else:
        raise ValueError("Invalid mode. Please use 'to' or 'la' for takeoff or landing mode.")
    
    if np.isnan(transition_index):
        if mode == 'la': 
            return np.nan, np.nan
        elif mode == 'to':
            _, la_index = tola_slope_detector(t[0:], h[0:], window_size, slope_threshold_takeoff, slope_threshold_landing, transition_alt, mode = 'la')
            return np.nan, la_index
    
    if mode == 'to':
        # Define window start and end
        t_window_end = transition_index
        t_window_start = max(0, transition_index - window_size)
    elif mode == 'la':
        t_window_start = transition_index
        t_window_end = min(len(t), transition_index + window_size)
    
    # Extract the time and altitude data for the window
    t_window = t[t_window_start:t_window_end]
    h_window = h[t_window_start:t_window_end]

    if t_window_end - t_window_start < 2:
        if mode == 'to':
            _, la_index = tola_slope_detector(t[transition_index:], h[transition_index:], window_size, slope_threshold_takeoff, slope_threshold_landing, transition_alt, mode = 'la')
            return np.nan, la_index
        elif mode == 'la':
            return np.nan, np.nan
        
    # Calculate the slope of the linear regression line
    slope, _, _, _, _ = stats.linregress(t_window, h_window)

    to_index = np.nan
    la_index = np.nan

    if mode == 'to':
        # Check if the transition at transition_index is greater than transition_alt_threshold
        # If the minimum altitude is 2000ft or something, then this is not a takeoff, we just detect landing and quit
        if h[transition_index] > transition_alt + transition_alt_threshold:
            # transition_index = np.nan
            _, la_index = tola_slope_detector(t[transition_index:], h[transition_index:], window_size, slope_threshold_takeoff, slope_threshold_landing, transition_alt, mode = 'la')
            return np.nan, la_index
        
    if mode == 'to':
        if slope >= slope_threshold_takeoff:
            # Find the index of the smallest positive altitude before transition_index
            min_index = np.where(h[:transition_index] > 0)[0][0]
            to_index = min_index 
            # if this is a takeoff, we need to run the landing detection on the rest of the data
            _, la_index = tola_slope_detector(t[transition_index:], h[transition_index:], window_size, slope_threshold_takeoff, slope_threshold_landing, transition_alt, mode = 'la')
            la_index = la_index + transition_index
        else:
            to_index = np.nan
            _, la_index = tola_slope_detector(t[transition_index:], h[transition_index:], window_size, slope_threshold_takeoff, slope_threshold_landing, transition_alt, mode = 'la')
            return np.nan, la_index
    
    elif mode == 'la':
        if slope < slope_threshold_landing:
            # Find the index of the smallest positive altitude before transition_index
            min_index = np.where(h[transition_index:] > 0)[0][-1]
            la_index = min_index + transition_index
        else:
            return np.nan, np.nan
    
    return to_index, la_index

df_airports = pd.read_csv(PATH_PREFIX + '/data/airports.csv')

def find_closest_airport(lat: float, lon: float):
    global df_airports # to prevent reloading the file due to multiple calls to find_closest_airport
    # Find the closest airport
    df_airports['distance'] = np.sqrt((df_airports['latitude'] - lat)**2 + (df_airports['longitude'] - lon)**2)
    closest_airport = df_airports.iloc[df_airports['distance'].idxmin()]
    error = closest_airport['distance']
    return closest_airport['icao'], closest_airport['latitude'], closest_airport['longitude'], error


def detect_takeoff_landing(sample_df: pd.DataFrame, to_file: str, la_file: str):
    # sample_df column names: id,lat,lon,velocity,heading,vertrate,baroaltitude,geoaltitude,lastposupdate
    # get all unique ids
    ids = sample_df['id'].unique()
    print('Processing ', len(ids), ' flights')

    h_distrib_to = np.zeros((len(ids),))
    h_distrib_la = np.zeros((len(ids),))


    # Create a CSV file writer
    to_writer = csv.writer(open(to_file, 'w'))
    la_writer = csv.writer(open(la_file, 'w'))
    # Write the header
    to_writer.writerow(['id','time','airport','lat','lon','error'])
    la_writer.writerow(['id','time','airport','lat','lon','error'])

    i = -1
    for id in tqdm(ids):
        try:
            i += 1
            df_id = sample_df[sample_df['id'] == id]
            t = df_id['lastposupdate'].values
            h = df_id['geoaltitude'].values
            lats = df_id['lat'].values
            lons = df_id['lon'].values
            ts, hs, latss, lonss = resample_data(t, h, lats, lons)
            to_index, la_index = tola_slope_detector(ts, hs)
            if not np.isnan(to_index):
                # Get the lat and lon of the takeoff and landing
                to_lat = latss[to_index]
                to_lon = lonss[to_index]
                # Find the closest airport
                to_airport, to_airport_lat, to_airport_lon, err = find_closest_airport(to_lat, to_lon)
                # Write the takeoff and landing to the CSV file
                to_writer.writerow([id, ts[to_index], to_airport, to_airport_lat, to_airport_lon, err])
                h_distrib_to[i] = hs[to_index]
            if not np.isnan(la_index):
                la_lat = latss[la_index]
                la_lon = lonss[la_index]
                # Find the closest airport
                la_airport, la_airport_lat, la_airport_lon, err = find_closest_airport(la_lat, la_lon)
                # Write the takeoff and landing to the CSV file
                la_writer.writerow([id, ts[la_index], la_airport, la_airport_lat, la_airport_lon, err])
                h_distrib_la[i] = hs[la_index]
        except Exception as e:
            print(f"Error processing flight with id: {id}")
            print(f"Error message: {str(e)}")
            # raise e
            continue

    return h_distrib_to, h_distrib_la


if __name__ == "__main__":
    sample_df = pd.read_csv(PATH_PREFIX + '/data/sample/sample.csv')
    print('Processing sample.csv')
    h_distrib_to, h_distrib_la = detect_takeoff_landing(sample_df, PATH_PREFIX + '/data/wp/sample.to.csv', PATH_PREFIX + '/data/wp/sample.la.csv')
    plt.hist(h_distrib_to, bins=100, alpha=0.5, label='Takeoff')
    plt.hist(h_distrib_la, bins=100, alpha=0.5, label='Landing')
    plt.legend(loc='upper right')
    plt.show()