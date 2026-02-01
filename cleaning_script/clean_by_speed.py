def clean_trajectory(df, max_speed_kmh=1000, min_speed_kmh=100):
    """
    Clean aircraft trajectory by removing unrealistic jumps.
    
    Parameters:
    - df: DataFrame with columns ['time', 'lat', 'lon']
    - max_speed_kmh: Maximum realistic speed in km/h
    - min_speed_kmh: Minimum realistic speed in km/h
    
    Returns:
    - Cleaned DataFrame
    """
    import numpy as np

    df = df.copy().sort_values('time')
    if len(df) < 2:
        return df

    lat = df['lat'].to_numpy()
    lon = df['lon'].to_numpy()
    time_vals = df['time'].to_numpy()

    # Vectorized haversine (km) between consecutive points.
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = 6371.0 * c

    time_diff_hours = (time_vals[1:] - time_vals[:-1]) / 3600.0
    speeds = np.divide(distance_km, time_diff_hours, out=np.full_like(distance_km, np.inf), where=time_diff_hours > 0)

    invalid = (speeds > max_speed_kmh) | (speeds < min_speed_kmh)
    if not np.any(invalid):
        return df

    to_drop = np.nonzero(invalid)[0] + 1
    return df.drop(df.index[to_drop])
