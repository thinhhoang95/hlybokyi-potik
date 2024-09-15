from matplotlib import pyplot as plt
import pandas as pd
import os
import sys
import pickle
from collections import deque
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Ensure that the script can locate the necessary modules
current_dir = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'turning_scripts'))

# Import custom modules
from data_preambles import dtypes_no_id, col_names, csv_to_exclude, catalog_col_names
from path_prefix import PATH_PREFIX
from get_turns import get_turning_points, TurnAndRise

def process_csv(csv_file):
    """
    Process a single CSV file: load data, compute turning points, and save segments.
    """
    try:
        # Define the path to the CSV file
        csv_path = os.path.join(PATH_PREFIX, 'data', 'csv', csv_file)
        
        # Load the CSV file into a DataFrame
        hour_df = pd.read_csv(csv_path, dtype=dtypes_no_id, parse_dates=True)
        hour_df.columns = col_names
        
        # Create a unique 'id' column by concatenating 'icao24' and 'callsign'
        hour_df['id'] = hour_df['icao24'] + hour_df['callsign']
        hour_ids = hour_df['id'].unique()
        print(f'There are {len(hour_ids)} unique ids in the CSV file {csv_file}')
    
        # Initialize deques to store segment data
        seg_from_lat = deque()
        seg_from_lon = deque()
        seg_to_lat = deque()
        seg_to_lon = deque()
    
        callsigns_skipped = 0
        
        # Iterate over each unique id to compute turning points
        for id in hour_ids:
            try:
                df_id = hour_df[hour_df['id'] == id]
                tr = get_turning_points(df_id)
            except ValueError as e:
                # If a ValueError occurs, skip this id
                callsigns_skipped += 1
                continue
    
            # Extract segments between turning points
            for i in range(len(tr['tp_time']) - 1):
                seg_from_lat.append(tr['tp_lat'][i])
                seg_from_lon.append(tr['tp_lon'][i])
                seg_to_lat.append(tr['tp_lat'][i+1])
                seg_to_lon.append(tr['tp_lon'][i+1])
    
        print(f'There were {len(hour_ids)} callsigns in {csv_file}, of which {callsigns_skipped} were skipped')
    
        # Extract timestamp from the filename
        timestamp = os.path.basename(csv_file).split('.')[0]
    
        # Prepare the segment data for pickling
        segment_data = {
            'seg_from_lat': list(seg_from_lat),
            'seg_from_lon': list(seg_from_lon),
            'seg_to_lat': list(seg_to_lat),
            'seg_to_lon': list(seg_to_lon)
        }
    
        # Ensure the segments directory exists
        segments_dir = os.path.join(PATH_PREFIX, 'data', 'segments')
        os.makedirs(segments_dir, exist_ok=True)
    
        # Define the path for the pickle file
        pickle_path = os.path.join(segments_dir, f'flight_segments_{timestamp}.segments.pickle')
        
        # Save the segment data to a pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(segment_data, f)
    
        print(f"Segment data saved to {pickle_path}")
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def main():
    """
    Main function to set up multiprocessing and process all CSV files.
    """
    # Define the directory containing CSV files
    csv_dir = os.path.join(PATH_PREFIX, 'data', 'csv')
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    # But not the ones with a dot in the beginning
    csv_files = [f for f in csv_files if not f.startswith('.')]
    
    # Determine the number of processes to use (e.g., number of CPU cores)
    num_processes = cpu_count()
    print(f"Starting processing with {num_processes} processes...")
    
    # Initialize a multiprocessing Pool
    with Pool(processes=num_processes) as pool:
        # Use tqdm to display a progress bar
        for _ in tqdm(pool.imap_unordered(process_csv, csv_files), total=len(csv_files)):
            pass  # The progress bar updates as each file is processed
    
    print("All files have been processed.")

if __name__ == '__main__':
    main()
