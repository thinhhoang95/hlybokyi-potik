# ************* PARAMETERS *************
ENABLE_SAMPLING = False
SAMPLE_SIZE = 5000
# ***************************************

from collections import deque
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

from cleaning_script import clean_by_speed as cleaner

# List all the csv files in the data/csv directory
csv_files = [f for f in os.listdir(f'{PATH_PREFIX}/data/csv') if f.endswith('.csv')]
# Remove files that start with underscore
csv_files = [f for f in csv_files if not f.startswith('._')]
# Remove files that contain any non-numeric characters except period and csv extension
csv_files = [f for f in csv_files if all(c.isdigit() or c == '.' or c in 'csv' for c in f)]


# Add MPIRE import at the top with other imports
from mpire import WorkerPool

from get_turns import get_turning_points
from collections import deque
from tqdm import tqdm

# Add worker function before the main loop
def process_csv_file(csv_file):
    print(f'Processing {csv_file}')
    # Check if result file already exists
    timestamp = int(csv_file.split('.')[0])
    result_file = f'{PATH_PREFIX}/data/hourly/{csv_file.split(".")[0]}.csv'
    if os.path.exists(result_file):
        print(f'Skipping {csv_file} - result already exists')
        return None
    
    # Load one CSV file
    hour_df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{csv_file}', dtype=dtypes_no_id, parse_dates=True)
    hour_df.columns = col_names
    hour_df['id'] = hour_df['icao24'] + hour_df['callsign']

    hour_ids = hour_df['id'].unique()
    print(f'There are {len(hour_ids)} unique ids in the hour_df')

    if ENABLE_SAMPLING:
        if len(hour_ids) > SAMPLE_SIZE:
            hour_ids = np.random.choice(hour_ids, SAMPLE_SIZE, replace=False)
            print(f'Sampled {len(hour_ids)} ids')
        else:
            print(f'Skipping sampling because there are less than {SAMPLE_SIZE} ids')

    seg_id = deque()
    seg_from_time = deque()
    seg_to_time = deque()
    seg_from_lat = deque()
    seg_from_lon = deque()
    seg_to_lat = deque()
    seg_to_lon = deque()
    seg_from_alt = deque()
    seg_to_alt = deque()
    seg_from_speed = deque()
    seg_to_speed = deque()

    callsigns_skipped = 0
    for id in hour_ids:
        try:
            df_id = hour_df[hour_df['id'] == id]
            # Clean the dataframe
            df_id = cleaner.clean_trajectory(df_id)
            tr = get_turning_points(df_id)
        except ValueError as e:
            callsigns_skipped += 1
            continue

        for i in range(len(tr['tp_time']) - 1):
            seg_id.append(id)
            seg_from_time.append(tr['tp_time'][i])
            seg_to_time.append(tr['tp_time'][i+1])
            seg_from_lat.append(tr['tp_lat'][i])
            seg_from_lon.append(tr['tp_lon'][i])
            seg_to_lat.append(tr['tp_lat'][i+1])
            seg_to_lon.append(tr['tp_lon'][i+1])
            seg_from_alt.append(tr['tp_alt'][i])
            seg_to_alt.append(tr['tp_alt'][i+1])
            seg_from_speed.append(tr['tp_vel'][i])
            seg_to_speed.append(tr['tp_vel'][i+1])

    print(f'There were {len(hour_ids)} callsigns, of which {callsigns_skipped} were skipped')
    
    segments_df = pd.DataFrame({
        'id': list(seg_id),
        'from_time': list(seg_from_time),
        'to_time': list(seg_to_time),
        'from_lat': list(seg_from_lat),
        'from_lon': list(seg_from_lon), 
        'to_lat': list(seg_to_lat),
        'to_lon': list(seg_to_lon),
        'from_alt': list(seg_from_alt),
        'to_alt': list(seg_to_alt),
        'from_speed': list(seg_from_speed),
        'to_speed': list(seg_to_speed)
    })
    
    output_filename = f'{PATH_PREFIX}/data/hourly/{csv_file.split(".")[0]}.csv'
    segments_df.to_csv(output_filename, index=False)
    print(f'Saved {len(segments_df)} segments to {output_filename}')
    return output_filename

# Replace the main loop with MPIRE processing
if __name__ == '__main__':
    # Use max_workers=None to automatically use all available CPU cores
    with WorkerPool(n_jobs=None) as pool:
        results = pool.map(process_csv_file, csv_files[:5], progress_bar=True)
    
    # Filter out None results (skipped files)
    completed_files = [r for r in results if r is not None]
    print(f'Processed {len(completed_files)} files successfully')
