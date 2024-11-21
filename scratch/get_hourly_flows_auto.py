# ************* PARAMETERS *************
ENABLE_SAMPLING = True
SAMPLE_SIZE = 2000
# ***************************************

from matplotlib import pyplot as plt
import pandas as pd

import os
import sys
import numpy as np
# Get the current directory
current_dir = os.path.dirname(os.path.abspath('__file__'))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'turning_scripts'))

from data_preambles import dtypes_no_id, col_names, csv_to_exclude, catalog_col_names
from path_prefix import PATH_PREFIX

# List all the csv files in the data/csv directory
csv_files = [f for f in os.listdir(f'{PATH_PREFIX}/data/csv') if f.endswith('.csv')]
# Remove files that start with underscore
csv_files = [f for f in csv_files if not f.startswith('._')]


for csv_file in csv_files[:10]:
    print('-'*100)
    print(f'Processing {csv_file}')
    # Check if result file already exists
    timestamp = int(csv_file.split('.')[0])
    result_file = f'{PATH_PREFIX}/data/hourly/{csv_file.split(".")[0]}_{timestamp}.csv'
    if os.path.exists(result_file):
        print(f'Skipping {csv_file} - result already exists')
        continue
    # Load one CSV file
    hour_df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{csv_file}', dtype=dtypes_no_id, parse_dates=True)
    hour_df.columns = col_names
    hour_df.head(5)
    # Add an id column
    hour_df['id'] = hour_df['icao24'] + hour_df['callsign']
    # hour_df.head(5)

    hour_ids = hour_df['id'].unique()
    print(f'There are {len(hour_ids)} unique ids in the hour_df')

    if ENABLE_SAMPLING:
        # Sample 1000 ids
        if len(hour_ids) > SAMPLE_SIZE:
            hour_ids = np.random.choice(hour_ids, SAMPLE_SIZE, replace=False)
            print(f'Sampled {len(hour_ids)} ids')
        else:
            print(f'Skipping sampling because there are less than {SAMPLE_SIZE} ids')

    from get_turns import get_turning_points, TurnAndRise
    from collections import deque
    from tqdm import tqdm

    # Note that get_turning_points will automatically add the first and last point to the list of turning points
    # So we don't need to add them manually

    # For each callsign, we attempt to get the turning points
    print('Creating segments...')
    print('Caution: will skip N/A rows')

    seg_id = deque()
    seg_time = deque()
    seg_from_lat = deque()
    seg_from_lon = deque()
    seg_to_lat = deque()
    seg_to_lon = deque()

    callsigns_skipped = 0
    for id in tqdm(hour_ids):
        try:
            df_id = hour_df[hour_df['id'] == id]
            tr = get_turning_points(df_id)
        except ValueError as e:
            # print(f'Skipping {id} because {e}')
            callsigns_skipped += 1
            continue

        # For each turn, we get the segment from and to
        for i in range(len(tr['tp_time']) - 1):
            seg_id.append(id)
            seg_time.append(tr['tp_time'][i])
            seg_from_lat.append(tr['tp_lat'][i])
            seg_from_lon.append(tr['tp_lon'][i])
            seg_to_lat.append(tr['tp_lat'][i+1])
            seg_to_lon.append(tr['tp_lon'][i+1])

    print(f'There were {len(hour_ids)} callsigns, of which {callsigns_skipped} were skipped')
        
    from datetime import datetime, timezone

    timestamp = int(csv_file.split('.')[0])

    # Convert timestamp to datetime object in UTC
    utc_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    # Format the datetime as a string
    formatted_datetime = utc_datetime.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Create a DataFrame with the segments
    import pandas as pd
    
    segments_df = pd.DataFrame({
        'id': list(seg_id),
        'time': list(seg_time),
        'from_lat': list(seg_from_lat),
        'from_lon': list(seg_from_lon), 
        'to_lat': list(seg_to_lat),
        'to_lon': list(seg_to_lon)
    })
    
    # Save to CSV with timestamp in filename
    output_filename = f'{PATH_PREFIX}/data/hourly/{csv_file.split(".")[0]}_{timestamp}.csv'
    segments_df.to_csv(output_filename, index=False)
    print(f'Saved {len(segments_df)} segments to {output_filename}')
