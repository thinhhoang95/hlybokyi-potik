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
  python pre_scripts2/get_route.py -i summer24/raw/cs -o summer24/routes_fast -j 7 -p
"""
import argparse
import pandas as pd
import time

import os
# Add this near the top of your file, before any multiprocessing code
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import numpy as np

from turning_scripts.get_turns import get_turning_points

from data_preambles import dtypes_no_id
CSV_USECOLS = [0, 1, 2, 3, 4, 5, 6]
CSV_COL_NAMES = ['time', 'icao24', 'lat', 'lon', 'heading', 'callsign', 'geoaltitude']
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
    p.add_argument(
        '--jobs', '-j',
        type=int,
        default=1,
        help='Number of worker processes (default: 1)',
    )
    p.add_argument(
        '--progress', '-p',
        action='store_true',
        help='Show progress bars (file-level when --jobs=1, per-id for single file)',
    )
    return p.parse_args()


def process_csv_file(
    csv_file,
    output_dir,
    enable_sampling,
    sample_size,
    show_progress=True,
    progress_every=None,
):
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
    hour_df = pd.read_csv(
        csv_file,
        dtype=dtypes_no_id,
        usecols=CSV_USECOLS,
        names=CSV_COL_NAMES,
        header=0,
        memory_map=True,
    )
    hour_df['id'] = hour_df['icao24'].str.upper() + hour_df['callsign'].str.strip().str.upper()

    # Drop NaN ids early; they never match in the old `hour_df[hour_df['id'] == id]` loop anyway.
    hour_df = hour_df[hour_df['id'].notna()]
    if enable_sampling:
        hour_ids = hour_df['id'].unique()
        if len(hour_ids) > sample_size:
            hour_ids = np.random.choice(hour_ids, sample_size, replace=False)
            print(f'Sampled {len(hour_ids)} ids')
        else:
            print(f'Skipping sampling because there are less than {sample_size} ids')

        # Filter once, then group/iterate. This preserves exact per-id rows while avoiding
        # O(N_rows) boolean scans for every id.
        hour_df = hour_df[hour_df['id'].isin(hour_ids)]

    grouped = hour_df.groupby('id', sort=False)
    total_ids = grouped.ngroups
    print(f'There are {total_ids} unique ids in the hour_df')

    seg_id = []
    seg_from_time = []
    seg_to_time = []
    seg_from_lat = []
    seg_from_lon = []
    seg_to_lat = []
    seg_to_lon = []
    seg_from_alt = []
    seg_to_alt = []
    seg_from_speed = []
    seg_to_speed = []

    callsigns_skipped = 0

    progress_iter = grouped
    pbar = None
    if show_progress:
        pbar = tqdm(grouped, desc=f'Processing {file_basename}', total=total_ids)
        progress_iter = pbar

    start_time = time.perf_counter()
    processed_ids = 0
    pid = os.getpid()

    for id, df_id in progress_iter:
        try:
            # Clean the dataframe
            df_id = cleaner.clean_trajectory(df_id)
            tr = get_turning_points(df_id)
        except ValueError as e:
            callsigns_skipped += 1
            if pbar is not None:
                pbar.set_postfix(skipped=callsigns_skipped)
            continue
        finally:
            processed_ids += 1
            if progress_every and processed_ids % progress_every == 0:
                elapsed = max(time.perf_counter() - start_time, 1e-9)
                avg_speed = processed_ids / elapsed
                print(
                    f'[pid {pid}] {processed_ids}/{total_ids} processed, '
                    f'avg speed: {avg_speed:.1f}/s ({file_basename})',
                    flush=True,
                )

        tp_time = tr['tp_time']
        if len(tp_time) < 2:
            continue

        tp_lat = tr['tp_lat']
        tp_lon = tr['tp_lon']
        tp_alt = tr['tp_alt']
        tp_vel = tr['tp_vel']
        count = len(tp_time) - 1

        seg_id.extend([id] * count)
        seg_from_time.extend(tp_time[:-1])
        seg_to_time.extend(tp_time[1:])
        seg_from_lat.extend(tp_lat[:-1])
        seg_from_lon.extend(tp_lon[:-1])
        seg_to_lat.extend(tp_lat[1:])
        seg_to_lon.extend(tp_lon[1:])
        seg_from_alt.extend(tp_alt[:-1])
        seg_to_alt.extend(tp_alt[1:])
        seg_from_speed.extend(tp_vel[:-1])
        seg_to_speed.extend(tp_vel[1:])

    if pbar is not None:
        pbar.close()

    print(f'There were {total_ids} callsigns, of which {callsigns_skipped} were skipped')
    
    segments_df = pd.DataFrame({
        'id': seg_id,
        'from_time': seg_from_time,
        'to_time': seg_to_time,
        'from_lat': seg_from_lat,
        'from_lon': seg_from_lon,
        'to_lat': seg_to_lat,
        'to_lon': seg_to_lon,
        'from_alt': seg_from_alt,
        'to_alt': seg_to_alt,
        'from_speed': seg_from_speed,
        'to_speed': seg_to_speed
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
    jobs = args.jobs
    show_progress = args.progress

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
        return process_csv_file(
            csv_file,
            output_dir,
            enable_sampling,
            sample_size,
            show_progress=False,
            progress_every=100,
        )

    if jobs <= 1:
        if show_progress and len(csv_files) > 1:
            file_iter = tqdm(csv_files, desc='Processing files')
        else:
            file_iter = csv_files

        results = []
        for csv_file in file_iter:
            per_id_progress = show_progress and len(csv_files) == 1
            results.append(
                process_csv_file(
                    csv_file,
                    output_dir,
                    enable_sampling,
                    sample_size,
                    show_progress=per_id_progress,
                )
            )
        completed_files = [r for r in results if r is not None]
        print(f'Processed {len(completed_files)} files successfully')
    else:
        with WorkerPool(n_jobs=jobs) as pool:
            results = pool.map(process_one, csv_files, progress_bar=show_progress)
        completed_files = [r for r in results if r is not None]
        print(f'Processed {len(completed_files)} files successfully')


if __name__ == '__main__':
    main()
