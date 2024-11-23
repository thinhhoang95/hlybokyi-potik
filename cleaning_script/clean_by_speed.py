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
    from haversine import haversine
    import numpy as np
    
    df = df.copy().sort_values('time')
    
    # Calculate speed between consecutive points
    speeds = []
    to_drop = []
    
    for i in range(len(df)-1):
        point1 = (df.iloc[i]['lat'], df.iloc[i]['lon'])
        point2 = (df.iloc[i+1]['lat'], df.iloc[i+1]['lon'])
        time_diff = (df.iloc[i+1]['time'] - df.iloc[i]['time']) / 3600  # Convert to hours
        
        # Calculate distance in km
        distance = haversine(point1, point2)
        
        # Calculate speed in km/h
        speed = distance / time_diff if time_diff > 0 else float('inf')
        speeds.append(speed)
        
        # Mark points for removal if speed is unrealistic
        if speed > max_speed_kmh or speed < min_speed_kmh:
            # Mark the point that creates the jump
            to_drop.append(i+1)
    
    # Remove marked points
    clean_df = df.drop(df.index[to_drop])
    
    return clean_df