import numpy as np
import pandas as pd
from geo.drift_compensation import get_track_drift_rate, great_circle_distance
from changepy import pelt
from changepy.costs import normal_mean
import matplotlib.pyplot as plt

from typing import TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import zarr

def forward_fill(arr):
    """
    Forward fill missing values in an array.

    Parameters:
    arr (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: The array with missing values forward filled.
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

class TurnAndRise(TypedDict):
    # Direction change
    tp_time: np.ndarray
    tp_lat: np.ndarray
    tp_lon: np.ndarray
    tp_alt: np.ndarray
    landed: bool # 1 if the changepoint is the moment the aircraft landed (alt < 500)
    tp_wp: NotRequired[list] # list of waypoint names
    
    # Altitude change
    dp_time: np.ndarray
    dp_lat: np.ndarray
    dp_lon: np.ndarray
    dp_alt: np.ndarray

    # Flight identification
    ident: str

def take_off_detection(alt: np.ndarray, thr: float = 500) -> int:
    """
    Detects the time step when the aircraft takes off based on altitude data.

    Parameters:
    alt (np.ndarray): Array of altitude values.
    thr (float): Altitude threshold for detecting take-off. Default is 500.

    Returns:
    int: The time step when the aircraft takes off. Returns -1 if no take-off is detected.
    """

    t_above_thr = np.where(alt > thr, 1, 0) # Find the first time the altitude is above the threshold
    t_above_thr_diff = np.diff(t_above_thr) # Find the difference between consecutive time steps when the altitude is above the threshold
    if len(t_above_thr) == 0:
        return -1 # The aircraft never goes above the threshold so it is on the ground and this is not a take-off
    # Take off is the moment when the aircraft goes above the threshold 
    t_takeoff = np.where(t_above_thr_diff > 0.5)[0]
    if len(t_takeoff) == 0:
        return -1
    return t_takeoff[0] + 1 # Return the time step when the aircraft takes off

def landing_detection(alt: np.ndarray, thr: float = 500) -> int:
    """
    Detects the time step when the aircraft lands based on altitude data.

    Parameters:
    alt (np.ndarray): Array of altitude values.
    thr (float): Altitude threshold for detecting take-off. Default is 500.

    Returns:
    int: The time step when the aircraft lands. Returns -1 if no landing is detected.
    """

    t_above_thr = np.where(alt > thr, 1, 0) # Find the first time the altitude is above the threshold
    t_above_thr_diff = np.diff(t_above_thr) # Find the difference between consecutive time steps when the altitude is above the threshold
    if len(t_above_thr) == 0:
        return -1 # The aircraft never goes above the threshold so it is on the ground and this is not a take-off
    # Take off is the moment when the aircraft goes above the threshold 
    t_landing = np.where(t_above_thr_diff < -0.5)[0]
    if len(t_landing) == 0:
        return -1
    return t_landing[0] + 1 # Return the time step when the aircraft takes off

def get_turning_points(df_ident: pd.DataFrame) -> TurnAndRise:
    """
    Detects turning points in a flight trajectory based on the provided dataframe. Notice that the last moment for not yet landed flight is NOT considered a turning point.

    Args:
        df_ident (pd.DataFrame): The dataframe of a single identified flight.

    Returns:
        dict: A dictionary containing the turning points and altitude change points.

    Raises:
        None

    """
    # Drop all rows with NaN values
    df_ident = df_ident.dropna() 
    # If the dataframe is empty, we raise an error
    if len(df_ident) == 0:
        raise ValueError('The dataframe is empty.')
    # Extract the values from the dataframe
    rlastposupdate = forward_fill(df_ident['time'].values) # - df_ident['time'].min() # in miliseconds
    hdg = forward_fill(df_ident['heading'].values)
    lat = forward_fill(df_ident['lat'].values)
    lon = forward_fill(df_ident['lon'].values)
    alt = forward_fill(df_ident['geoaltitude'].values)
    ident = df_ident['id'].values[0]
    vel = np.zeros_like(hdg)

    # Detection of turning points
    # Compute the drift compensation
    track_drift = np.zeros_like(hdg)
    cumul_drift = 0
    hdg_compensated = np.zeros_like(hdg)
    for i in range(1, len(hdg)):
        # Estimate the velocity of the aircraft
        vel[i-1] = great_circle_distance(lat[i-1], lon[i-1], lat[i], lon[i]) / (rlastposupdate[i] - rlastposupdate[i-1]) # in km/s
        # We will use the last time's value to compensate the drift for this time
        track_drift[i] = get_track_drift_rate(lat[i-1], lon[i-1], hdg[i-1]) * vel[i-1] * (rlastposupdate[i] - rlastposupdate[i-1])
        cumul_drift += track_drift[i]
        hdg_compensated[i] = (hdg[i] - cumul_drift) % 360

    # Get the changepoints
    changepoints = pelt(normal_mean(hdg_compensated, 1), len(hdg_compensated))

    # Write down the turning points
    tp_lat = []
    tp_lon = []
    tp_time = []
    tp_alt = []
    tp_vel = []

    flight_not_landed_yet = True

    # Detection of takeoffs
    t_takeoff = take_off_detection(alt)
    t_landing = landing_detection(alt)

    # One final changepoint at the end of the flight or when the aircraft lands
    if t_landing != -1:
        changepoints = np.append(changepoints, t_landing)
        # Delete all the changepoints after the aircraft landed
        changepoints = changepoints[changepoints <= t_landing] 
        flight_not_landed_yet = False
    else:
        changepoints = np.append(changepoints, len(hdg_compensated)-1)

    if t_takeoff != -1:
        changepoints = np.insert(changepoints, 0, t_takeoff)
        # Delete all the changepoints before the aircraft took off
        changepoints = changepoints[changepoints >= t_takeoff]

    for i in range(len(changepoints)):
        tp_lat.append(lat[changepoints[i]])
        tp_lon.append(lon[changepoints[i]])
        tp_time.append(rlastposupdate[changepoints[i]])
        tp_alt.append(alt[changepoints[i]])
        tp_vel.append(vel[changepoints[i]])

    # The beginning and the end of the flight are also changepoints
    tp_lat.insert(0, lat[0])
    tp_lon.insert(0, lon[0])
    tp_time.insert(0, rlastposupdate[0])
    tp_alt.insert(0, alt[0])
    tp_vel.insert(0, vel[0])
    changepoints = np.insert(changepoints, 0, 0)

    tp_lat.append(lat[-1])
    tp_lon.append(lon[-1])
    tp_time.append(rlastposupdate[-1])
    tp_alt.append(alt[-1])
    tp_vel.append(vel[-1])
    changepoints = np.append(changepoints, len(hdg_compensated)-1)
    
    # Merge changepoints that are too close to each other
    i = 0
    while i < len(tp_lat)-1:
        # print(i, tp_time[i], tp_time[i+1])
        if (tp_time[i+1] - tp_time[i]) < 60:
            # print(f'Merging {i} and {i+1}')
            tp_lat[i] = (tp_lat[i] + tp_lat[i+1]) / 2
            tp_lon[i] = (tp_lon[i] + tp_lon[i+1]) / 2
            tp_time[i] = (tp_time[i] + tp_time[i+1]) / 2
            tp_alt[i] = (tp_alt[i] + tp_alt[i+1]) / 2
            tp_vel[i] = (tp_vel[i] + tp_vel[i+1]) / 2
            tp_lat.pop(i+1)
            tp_lon.pop(i+1)
            tp_time.pop(i+1)
            tp_alt.pop(i+1)
            tp_vel.pop(i+1)
        else:
            i += 1

    # Add takeoff to the changepoints
    if t_takeoff != -1:
        if len(tp_time) > 0:
            if tp_time[0] > rlastposupdate[t_takeoff]:
                # print('Adding takeoff changepoint')
                tp_lat.insert(0, lat[t_takeoff])
                tp_lon.insert(0, lon[t_takeoff])
                tp_time.insert(0, rlastposupdate[t_takeoff])
                tp_alt.insert(0, alt[t_takeoff])
                tp_vel.insert(0, vel[t_takeoff])
                
    # Add landing to the changepoints
    if t_landing != -1:
        if len(tp_time) > 0:
            if tp_time[-1] < rlastposupdate[t_landing]:
                # print('Adding landing changepoint')
                tp_lat.append(lat[t_landing])
                tp_lon.append(lon[t_landing])
                tp_time.append(rlastposupdate[t_landing])
                tp_alt.append(alt[t_landing])
                tp_vel.append(vel[t_landing])
    result_turn = {
        'tp_time': np.array(tp_time),
        'tp_lat': np.array(tp_lat),
        'tp_lon': np.array(tp_lon),
        'tp_alt': np.array(tp_alt),
        'tp_vel': np.array(tp_vel),
        'landed': not flight_not_landed_yet,
        'ident': ident
    }

    result_alt = get_altitude_change_points(rlastposupdate, lat, lon, alt, vel)

    return {
        **result_turn,
        **result_alt
    }

def get_altitude_change_points(rlastposupdate: pd.DataFrame, lat: np.ndarray, lon: np.ndarray, alt:np.ndarray, vel:np.ndarray) -> TurnAndRise:
    """
    Get altitude change points based on the provided data.

    Args:
        rlastposupdate (pd.DataFrame): DataFrame containing the time information.
        lat (np.ndarray): Array of latitude values.
        lon (np.ndarray): Array of longitude values.
        alt (np.ndarray): Array of altitude values.

    Returns:
        dict: A dictionary containing the altitude change points with keys 'dp_time', 'dp_lat', 'dp_lon', and 'dp_alt'.
    """

    dp_time = []
    dp_lat = []
    dp_lon = []
    dp_alt = []
    dp_vel = []

    # Get the changepoints
    changepoints = pelt(normal_mean(alt, 1000), len(alt))

    # Add the initial and final changepoints
    changepoints = np.insert(changepoints, 0, 0)
    changepoints = np.append(changepoints, len(alt)-1)

    for i in range(len(changepoints)):
        dp_time.append(rlastposupdate[changepoints[i]])
        dp_lat.append(lat[changepoints[i]])
        dp_lon.append(lon[changepoints[i]])
        dp_alt.append(alt[changepoints[i]])
        dp_vel.append(vel[changepoints[i]])

    # Merge changepoints that are too close to each other
    i = 0
    while i < len(dp_lat)-1:
        if (dp_time[i+1] - dp_time[i]) < 120:
            dp_lat[i] = (dp_lat[i] + dp_lat[i+1]) / 2
            dp_lon[i] = (dp_lon[i] + dp_lon[i+1]) / 2
            dp_time[i] = (dp_time[i] + dp_time[i+1]) / 2
            dp_alt[i] = (dp_alt[i] + dp_alt[i+1]) / 2
            dp_vel[i] = (dp_vel[i] + dp_vel[i+1]) / 2
            dp_lat.pop(i+1)
            dp_lon.pop(i+1)
            dp_time.pop(i+1)
            dp_alt.pop(i+1)
            dp_vel.pop(i+1)
        else:
            i += 1

    return {
        'dp_time': np.array(dp_time),
        'dp_lat': np.array(dp_lat),
        'dp_lon': np.array(dp_lon),
        'dp_alt': np.array(dp_alt),
        'dp_vel': np.array(dp_vel)
    }

def plot_changepoints(tr: TurnAndRise, df: pd.DataFrame = None, ident:str = None) -> None:
    tp_lat = tr['tp_lat']
    tp_lon = tr['tp_lon']
    tp_alt = tr['tp_alt']
    tp_time = tr['tp_time']
    dp_lat = tr['dp_lat']
    dp_lon = tr['dp_lon']
    dp_alt = tr['dp_alt']
    dp_time = tr['dp_time']

    # Check if the dataframe has the required columns
    if 'ident' not in df.columns:
        df['ident'] = (df['callsign'].str.strip()+'_'+df['icao24'].str.strip())

    # If ident is specified, filter the dataframe for the specific ident
    if ident is not None:
        df_ident = df[df['ident'] == ident]
    else:
        df_ident = df 
    
    # Plot the changepoints
    plt.figure(figsize=(6,6))
    if df_ident is not None:
        plt.plot(df_ident['lon'], df_ident['lat'], 'black') # flight path
    plt.plot(tp_lon, tp_lat, 'go', markersize=5) # turning points
    plt.plot(dp_lon, dp_lat, 'rx', markersize=3) # altitude change points

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flight Path with Turning Points (Landed: {})'.format(tr['landed']))

def write_turnandrise_to_zarr(tr: TurnAndRise,  zarr_path: str) -> None:
    zarr_group = zarr.open(zarr_path, mode='w')

    # Save each item in the dictionary to the ZARR group
    for key, value in tr.items():
        if isinstance(value, np.ndarray):
            # Store numpy arrays directly
            zarr_group.create_dataset(key, data=value)
        elif isinstance(value, dict):
            # Store dictionaries as sub-groups with attributes
            sub_group = zarr_group.create_group(key)
            # print('key:', key, 'value:', value)
            for sub_key, sub_value in value.items():
                # print('sub_key:', sub_key, 'sub_value:', sub_value)
                sub_group.attrs[sub_key] = sub_value
        else:
            # Store other types of data as attributes
            # print('key:', key, 'value:', value)
            zarr_group.attrs[key] = value

def load_turnandrise_from_zarr(zarr_path: str) -> TurnAndRise:
    # Open the ZARR file in read mode
    zarr_group = zarr.open(zarr_path, mode='r')

    # Load the data into a dictionary
    loaded_dict = {}
    for key in zarr_group:
        # print('Key:', key)
        if isinstance(zarr_group[key], zarr.core.Array):
            # If it's an array, load it as a numpy array
            loaded_dict[key] = zarr_group[key][:]
            #print('Key:', key, 'Value:', loaded_dict[key])
        elif isinstance(zarr_group[key], zarr.hierarchy.Group):
            # If it's a group, load its attributes into a dictionary
            loaded_dict[key] = dict(zarr_group[key].attrs)
            #print('Key:', key, 'Value:', loaded_dict[key])
        
    # Load the attrs 
    for key in zarr_group.attrs:
        loaded_dict[key] = zarr_group.attrs[key]

    return loaded_dict

import io
from typing import TextIO
def get_turns_and_rise_from_files(turns_df:pd.DataFrame, rises_df:pd.DataFrame, ident: str) -> TurnAndRise:
    # rises_df column names: ident,time,lat,lon,alt,vel
    # turns_df column names: ident,wp,time,lat,lon,alt,vel
    # Filter the dataframes for the specific ident
    rises_df_ident = rises_df[rises_df['ident'] == ident]
    turns_df_ident = turns_df[turns_df['ident'] == ident]
    if len(rises_df_ident) == 0 or len(turns_df_ident) == 0:
        raise ValueError('No data found for the specified ident.')
    # Prepare the dictionary
    tr = {
        'tp_time': turns_df_ident['time'].values,
        'tp_lat': turns_df_ident['lat'].values,
        'tp_lon': turns_df_ident['lon'].values,
        'tp_alt': turns_df_ident['alt'].values,
        'tp_vel': turns_df_ident['vel'].values,
        'landed': False,
        'ident': ident,
        'dp_time': rises_df_ident['time'].values,
        'dp_lat': rises_df_ident['lat'].values,
        'dp_lon': rises_df_ident['lon'].values,
        'dp_alt': rises_df_ident['alt'].values,
        'dp_vel': rises_df_ident['vel'].values
    }
    # Landed is True if the last altitude is less than 1000
    if tr['dp_alt'][-1] < 1000:
        tr['landed'] = True
    if tr['tp_alt'][-1] < 1000:
        tr['landed'] = True

    # Add the wp column to the dictionary if it exists
    tr['tp_wp'] = turns_df_ident['wp'].tolist()

    return tr
    