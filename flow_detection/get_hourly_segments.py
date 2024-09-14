from matplotlib import pyplot as plt
import pandas as pd

import os
import sys

# Get the current directory
current_dir = os.path.dirname(os.path.abspath('__file__'))

# Add the parent directory to sys.path
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'turning_scripts'))

from data_preambles import dtypes_no_id, col_names, csv_to_exclude, catalog_col_names
from path_prefix import PATH_PREFIX

# List all CSV files in the data/csv directory
csv_files = [f for f in os.listdir(f'{PATH_PREFIX}/data/csv') if f.endswith('.csv')]
# Remove the files with a dot in the first character like .DS_Store
csv_files = [f for f in csv_files if not f.startswith('.')]

for csv_file in csv_files:
    # Load one CSV file
    hour_df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{csv_file}', dtype=dtypes_no_id, parse_dates=True)
    hour_df.columns = col_names
    # Add an id column
    hour_df['id'] = hour_df['icao24'] + hour_df['callsign']
    hour_ids = hour_df['id'].unique()
    print(f'There are {len(hour_ids)} unique ids in the CSV file {csv_file}')

    from get_turns import get_turning_points, TurnAndRise
    from collections import deque
    from tqdm import tqdm

    # Note that get_turning_points will automatically add the first and last point to the list of turning points
    # So we don't need to add them manually

    # For each callsign, we attempt to get the turning points
    print('Creating segments...')
    print('Caution: will skip N/A rows')

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
            seg_from_lat.append(tr['tp_lat'][i])
            seg_from_lon.append(tr['tp_lon'][i])
            seg_to_lat.append(tr['tp_lat'][i+1])
            seg_to_lon.append(tr['tp_lon'][i+1])

    print(f'There were {len(hour_ids)} callsigns, of which {callsigns_skipped} were skipped')

    # Get the timestamp from the filename
    timestamp = os.path.basename(csv_files[0]).split('.')[0]

    # Save the segments to a pickle file
    import pickle

    # Create a dictionary to store the segment data
    segment_data = {
        'seg_from_lat': seg_from_lat,
        'seg_from_lon': seg_from_lon,
        'seg_to_lat': seg_to_lat,
        'seg_to_lon': seg_to_lon
    }

    # Create a directory for the pickle files if it doesn't exist
    os.makedirs(f'{PATH_PREFIX}/data/segments', exist_ok=True)

    # Save the segment data to a pickle file
    with open(f'{PATH_PREFIX}/data/segments/flight_segments_{timestamp}.segments.pickle', 'wb') as f:
        pickle.dump(segment_data, f)

    print(f"Segment data saved to {PATH_PREFIX}/data/segments/flight_segments_{timestamp}.segments.pickle")
