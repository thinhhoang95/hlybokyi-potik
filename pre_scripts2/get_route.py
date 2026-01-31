"""
Compute route segments from CS state CSV files.

Sample usage:
  # Defaults: input = PATH_PREFIX/pre_scripts2/cs, output = routes
  python pre_scripts2/get_route.py

  # Custom input and output directories
  python pre_scripts2/get_route.py -i /path/to/cs -o /path/to/routes

  # With sampling (limit callsigns per file)
  python pre_scripts2/get_route.py --sample --sample-size 3000

  # Short flags
  python pre_scripts2/get_route.py -i summer24/raw/cs -o summer24/routes -s -n 5000
"""
from collections import deque
import argparse
import pandas as pd

import os
# Add this near the top of your file, before any multiprocessing code
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
import sys
import numpy as np

from turning_scripts.get_turns import get_turning_points

from data_preambles import dtypes_no_id
col_names = ['time', 'icao24', 'lat', 'lon', 'heading', 'callsign', 'geoaltitude', 'id']
from path_prefix import PATH_PREFIX

from cleaning_script import clean_by_speed as cleaner
from mpire import WorkerPool
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments for input/output dirs and optional sampling.

    Returns:
        argparse.Namespace: Parsed args with ``input_dir``, ``output_dir``,
        ``sample`` (bool), and ``sample_size`` (int).
    """
    default_cs_dir = os.path.join(PATH_PREFIX, 'pre_scripts2', 'cs')
    p = argparse.ArgumentParser(description='Compute route segments from CS state CSV files.')
    p.add_argument(
        '--input-dir', '-i',
        default=default_cs_dir,
        help=f'Input directory containing CS CSV files (default: {default_cs_dir})',
    )
    p.add_argument(
        '--output-dir', '-o',
        default='routes',
        help='Output directory for route segment CSVs (default: routes)',
    )
    p.add_argument(
        '--sample', '-s',
        action='store_true',
        help='Enable sampling of callsigns per file',
    )
    p.add_argument(
        '--sample-size', '-n',
        type=int,
        default=5000,
        help='Sample size when --sample is set (default: 5000)',
    )
    return p.parse_args()


def process_csv_file(csv_file, output_dir, enable_sampling, sample_size, show_progress=False):
    """Process one CS state CSV into a route-segments CSV.

    Loads the CSV, builds flight ``id`` as icao24 + callsign (uppercase). Optionally
    samples up to ``sample_size`` unique ids per file. For each id: cleans the
    trajectory with ``clean_by_speed.clean_trajectory``, gets turning points with
    ``get_turning_points``, then emits one segment per pair of consecutive turning
    points. Skips ids that raise ValueError (e.g. cleaning/turning logic). Writes
    one output CSV with the same basename as the input under ``output_dir``; skips
    processing if that output file already exists.

    Parameters
    ----------
    csv_file : str
        Path to the input CS state CSV.
    output_dir : str
        Directory where the route-segments CSV will be written.
    enable_sampling : bool
        If True, process at most ``sample_size`` randomly chosen ids per file.
    sample_size : int
        Max number of ids to process per file when ``enable_sampling`` is True.

    Returns
    -------
    str | None
        Path to the written segments CSV, or None if the file was skipped
        (output already existed) or no segments were produced.
    """
    print(f'Processing {csv_file}')
    file_basename = os.path.basename(csv_file)
    result_file = os.path.join(output_dir, f'{file_basename.split(".")[0]}.csv')
    if os.path.exists(result_file):
        print(f'Skipping {csv_file} - result already exists')
        return None
    
    # Load one CSV file
    hour_df = pd.read_csv(f'{csv_file}', dtype=dtypes_no_id, parse_dates=True)
    hour_df.columns = col_names
    hour_df['id'] = hour_df['icao24'].str.upper() + hour_df['callsign'].str.strip().str.upper()

    hour_ids = hour_df['id'].unique()
    print(f'There are {len(hour_ids)} unique ids in the hour_df')

    if enable_sampling:
        if len(hour_ids) > sample_size:
            hour_ids = np.random.choice(hour_ids, sample_size, replace=False)
            print(f'Sampled {len(hour_ids)} ids')
        else:
            print(f'Skipping sampling because there are less than {sample_size} ids')

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
    progress_iter = hour_ids
    pbar = None
    if show_progress:
        pbar = tqdm(hour_ids, desc=f'Processing {file_basename}', total=len(hour_ids))
        progress_iter = pbar

    for id in progress_iter:
        try:
            df_id = hour_df[hour_df['id'] == id]
            # Clean the dataframe
            df_id = cleaner.clean_trajectory(df_id)
            tr = get_turning_points(df_id)
        except ValueError as e:
            callsigns_skipped += 1
            if pbar is not None:
                pbar.set_postfix(skipped=callsigns_skipped)
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

    if pbar is not None:
        pbar.close()

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
    
    output_filename = os.path.join(output_dir, f'{file_basename.split(".")[0]}.csv')
    segments_df.to_csv(output_filename, index=False)
    print(f'Saved {len(segments_df)} segments to {output_filename}')
    return output_filename

def main():
    """Discover CS state CSVs in the input directory and process them into route segments.

    Creates the output directory if needed, walks the input dir for ``.csv`` files
    (excluding names starting with ``._``), and processes each file via
    ``process_csv_file``. Parallel processing (WorkerPool) is currently commented
    out; uncomment to run all files in parallel.
    """
    args = parse_args()
    cs_input_dir = args.input_dir
    output_dir = args.output_dir
    enable_sampling = args.sample
    sample_size = args.sample_size

    print(f'Sampling enabled with sample size of {sample_size}' if enable_sampling else 'Sampling disabled')

    csv_files = []
    for root, _dirs, files in os.walk(cs_input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('._')]

    print(f'Found {len(csv_files)} csv files')
    if not csv_files:
        print('No CSV files found. Exiting.')
        return
    print(f'Processing {len(csv_files)} csv files')

    os.makedirs(output_dir, exist_ok=True)

    def process_one(csv_file):
        return process_csv_file(csv_file, output_dir, enable_sampling, sample_size, show_progress=False)

    # Process first file only (uncomment below for parallel processing of all files)
    # process_csv_file(csv_files[0], output_dir, enable_sampling, sample_size, show_progress=True)
    with WorkerPool(n_jobs=None) as pool:
        results = pool.map(process_one, csv_files, progress_bar=True)
    completed_files = [r for r in results if r is not None]
    print(f'Processed {len(completed_files)} files successfully')


if __name__ == '__main__':
    main()
