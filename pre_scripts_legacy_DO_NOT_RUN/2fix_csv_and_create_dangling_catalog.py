# THIS CODE IS DEPRECATED. See `building_dataset.md` and follow the instructions from `building_dataset.md`

"""Fix CSV state files and build a catalog of "dangling" (spanning) flights.

This script processes hourly flight-state CSV files under ``data/csv/``: it
cleans each file (deduplicates rows, drops missing values, ensures an ``id``
column) and records which flight IDs appear in which files. From that it
builds a "dangling" catalog: flights that span multiple hourly files (i.e.
appear in more than one timestamp). The catalog is written as:

- ``data/csv/dangling.buffer.csv`` — raw (id, file) pairs from the first pass
- ``data/csv/dangling.csv`` — aggregated per flight instance: id, from_timestamp, to_timestamp
  where ``id`` is ``<callsign+icao24>|<segment_start_timestamp>`` (see "instance" rule below).
- ``data/csv/dangling_explicit.csv`` — same as dangling.csv plus an ``all_files``
  column listing every CSV file the instance appears in (derived from the buffer when available;
  does not assume the flight appears in every intermediate hour). Entries are separated
  by a backslash (``\\``) to avoid ambiguity with subdirectory paths.

Master entry points (set at top of script):
- ``CSV_INPUT_DIR``: directory from which hourly CSVs are read.
- ``CSV_OUTPUT_DIR``: directory where ``dangling.buffer.csv``, ``dangling.csv``,
  and ``dangling_explicit.csv`` are written. Defaults match ``PATH_PREFIX/data/csv``.

Other config:
- ``path_prefix.PATH_PREFIX``: used to build default input/output dirs.
- ``data_preambles.dtypes_no_id``, ``col_names``, ``csv_to_exclude``: CSV schema and exclusions.

Input (expected):
- ``CSV_INPUT_DIR`` containing hourly CSVs named by Unix timestamp, e.g.
  ``1711940400.csv`` (may be in subdirectories). Each CSV has no header;
  columns match ``col_names``.

Output (created/overwritten under ``CSV_OUTPUT_DIR``):
- ``dangling.buffer.csv``: header ``id,file``; one row per (flight id, file timestamp).
- ``dangling.csv``: header ``id,from_timestamp,to_timestamp``; one row per spanning flight instance.
- ``dangling_explicit.csv``: same as dangling.csv plus ``all_files`` (backslash-separated filenames).

Example (conceptual):

  Input files: 1711940400.csv, 1711944000.csv, 1711947600.csv
  Flight "ABC123_abc123" appears in 1711940400.csv and 1711944000.csv.

  After fix_csv_and_create_dangling_catalog() + finalize_dangling_catalog():
  - dangling.buffer.csv contains (among others): (ABC123_abc123, 1711940400),
    (ABC123_abc123, 1711944000).
  - dangling.csv contains: id=ABC123_abc123|1711940400, from_timestamp=1711940400,
    to_timestamp=1711944000.

  After build_explicit_dangling_catalog():
  - dangling_explicit.csv adds: all_files = "1711940400.csv\\1711944000.csv"
    (or paths with subdirs, e.g. "2024-04-01/1711940400.csv\\2024-04-01/1711944000.csv",
    if CSVs live in subdirectories).

CLI:
- No arguments: run full pipeline (fix CSV, build buffer, finalize dangling, build explicit).
- ``--only-explicit``: skip scanning/fixing CSVs; load existing ``dangling.csv`` from
  ``CSV_OUTPUT_DIR`` and build only ``dangling_explicit.csv``.
"""

import argparse
import sys
import os
import pandas as pd
import csv
import datetime
# Add the path prefix to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from path_prefix import PATH_PREFIX

from data_preambles import dtypes_no_id, csv_to_exclude, col_names

# Master entry points: change these to read/write from different locations.
CSV_INPUT_DIR = os.path.join(PATH_PREFIX, 'summer24', 'raw')
CSV_OUTPUT_DIR = os.path.join(PATH_PREFIX, 'summer24', 'dangling')

# Ensure the output directory exists before writing any files
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Output filenames (written under CSV_OUTPUT_DIR).
DANGLING_BUFFER_FILENAME = 'dangling.buffer.csv'
DANGLING_FILENAME = 'dangling.csv'
DANGLING_EXPLICIT_FILENAME = 'dangling_explicit.csv'

# Chunk size for building the explicit dangling catalog without high memory use.
EXPLICIT_CHUNK_SIZE = 5000
# Expected spacing (seconds) between hourly CSVs.
CSV_TIME_STEP_SECONDS = 3600
# Maximum gap (seconds) allowed between consecutive files for the same flight instance.
# If the gap is larger, we start a new instance of that (callsign+icao24) id.
INSTANCE_GAP_HOURS = 5
INSTANCE_GAP_SECONDS = INSTANCE_GAP_HOURS * 3600
# Instance id separator: instance_id = "<callsign+icao24>|<segment_start_timestamp>"
INSTANCE_ID_SEPARATOR = '|'
# Separator between file paths in dangling_explicit.csv (use backslash to avoid
# ambiguity with subdirectory paths).
ALL_FILES_SEPARATOR = '\\'


def list_csv_files_recursive(root_dir: str) -> list[str]:
    """List all .csv files under root_dir, returning paths relative to root_dir.

    Walks subdirectories recursively. Paths use os.sep. Excludes files whose
    basename is in csv_to_exclude (e.g. catalog.csv) or starts with '._' (macOS resource forks).
    """
    csv_files = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)
        if rel_dir == '.':
            rel_dir = ''
        for f in filenames:
            # Exclude files whose basename is in csv_to_exclude or starts with '._'
            if not f.endswith('.csv') or f in csv_to_exclude or f.startswith('._'):
                continue
            rel_path = os.path.join(rel_dir, f) if rel_dir else f
            csv_files.append(rel_path)
    return csv_files


def fix_csv_and_create_dangling_catalog() -> None:
    """Scan all hourly CSVs, clean each, and write a buffer catalog of (id, file) pairs.

    Recursively finds every CSV under ``CSV_INPUT_DIR`` (excluding basenames in
    ``csv_to_exclude``). Paths used elsewhere (e.g. ``all_files``) preserve subdirs.
    For each file: loads with ``dtypes_no_id`` and ``col_names``, adds an ``id``
    column (callsign + icao24) if missing, removes duplicate (id, time) rows and
    rows with missing values, then appends (id, timestamp) for each unique id to
    ``dangling.buffer.csv`` in ``CSV_OUTPUT_DIR``. Does not overwrite the original CSVs.
    """
    # # If catalog.csv exists, read it
    # if os.path.exists(f'{PATH_PREFIX}/data/csv/catalog.csv'):
    #     spanning_df = pd.read_csv(f'{PATH_PREFIX}/data/csv/catalog.csv')
    # else:
    spanning_buffer_file = os.path.join(CSV_OUTPUT_DIR, DANGLING_BUFFER_FILENAME)
    with open(spanning_buffer_file, 'w', newline='') as f_out:
        spanning_buffer_writer = csv.writer(f_out)
        spanning_buffer_writer.writerow(['id', 'file'])

        csv_files = list_csv_files_recursive(CSV_INPUT_DIR)
        print(f'Found {len(csv_files)} csv files in {CSV_INPUT_DIR}')

        # Sort the csv_files list
        csv_files.sort()

        # For debugging, just take the first 10 csv files
        # csv_files = csv_files[:20]

        # For each csv.gz file, read the csv.gz file and add an id column
        print('Beginning to process CSV files... ')
        print('Notice: Warnings are ignored')
        import warnings
        warnings.filterwarnings('ignore')
        for file_id, csv_file in tqdm(enumerate(csv_files)):
            # print(f'Processing {csv_file}')
            # print(f'Applying CSV fixes: removing duplicated rows and missing values...')
            try:
                df = pd.read_csv(os.path.join(CSV_INPUT_DIR, csv_file), header=None, dtype=dtypes_no_id)
            except Exception as e:
                print(f"Error skipping {csv_file}: {e}", file=sys.stderr)
                continue

            df.columns = col_names

            # Add header to the df if it doesn't have it
            if 'id' not in df.columns:
                df['id'] = df['callsign'] + df['icao24']

            # remove duplicated rows
            df = remove_duplicated_rows(df)
            # remove rows with missing values
            df = remove_rows_with_missing_values(df)
            # print('CSV fixes applied. Now handling spanning flights...')

            # Handle spanning flights
            this_csv_ids = df['id'].unique()
            # Timestamp from basename so subdir paths (e.g. 2024-04-01/1711940400.csv) work
            timestamp_of_csv_file = int(os.path.basename(csv_file).replace('.csv', ''))
            spanning_buffer_writer.writerows([[id, timestamp_of_csv_file] for id in this_csv_ids])

            # Save the fixed CSV file (not related to dangling flights, just some fixing of format and adding a header row)
            # df.to_csv(f'{PATH_PREFIX}/data/csv/{csv_file}', index=False)


def finalize_dangling_catalog() -> pd.DataFrame:
    """Aggregate the buffer catalog into dangling.csv and return the dataframe.

    Reads ``dangling.buffer.csv`` from ``CSV_OUTPUT_DIR``, splits each (callsign+icao24)
    id into *instances* using a time-gap rule, and writes ``dangling.csv`` with
    columns id, from_timestamp, to_timestamp. The output ``id`` is an instance id:
    ``<callsign+icao24>|<segment_start_timestamp>``.

    A new instance starts when the gap between consecutive files exceeds
    ``INSTANCE_GAP_SECONDS`` (5 hours by default). This avoids joining unrelated
    flights that reuse the same callsign+icao24 on different days.

    Rows where an id appears in only 1 file are excluded.
    """
    print('Finalizing the dangling catalog... This may take a while')

    spanning_buffer_df = pd.read_csv(os.path.join(CSV_OUTPUT_DIR, DANGLING_BUFFER_FILENAME))
    if spanning_buffer_df.empty:
        dangling_df = pd.DataFrame(columns=['id', 'from_timestamp', 'to_timestamp'])
        dangling_df.to_csv(os.path.join(CSV_OUTPUT_DIR, DANGLING_FILENAME), index=False)
        print('Created dangling_df with 0 rows')
        return dangling_df

    spanning_buffer_df = spanning_buffer_df[['id', 'file']].dropna()
    spanning_buffer_df['file'] = spanning_buffer_df['file'].astype('int64')
    spanning_buffer_df = spanning_buffer_df.drop_duplicates(subset=['id', 'file'])
    spanning_buffer_df = spanning_buffer_df.sort_values(['id', 'file'], kind='mergesort').reset_index(drop=True)

    # Split each id into instances: start a new instance if the gap is > INSTANCE_GAP_SECONDS.
    file_diff = spanning_buffer_df.groupby('id')['file'].diff()
    is_break = file_diff.isna() | (file_diff > INSTANCE_GAP_SECONDS)
    segment = is_break.groupby(spanning_buffer_df['id']).cumsum()
    spanning_buffer_df['segment'] = segment

    segments = (
        spanning_buffer_df.groupby(['id', 'segment'])['file']
        .agg(from_timestamp='min', to_timestamp='max', n_files='count')
        .reset_index()
    )

    # Keep only instances that appear in 2+ files.
    segments = segments[segments['n_files'] >= 2].copy()
    segments['id'] = segments['id'].astype(str) + INSTANCE_ID_SEPARATOR + segments['from_timestamp'].astype(str)
    dangling_df = segments[['id', 'from_timestamp', 'to_timestamp']].sort_values(
        ['id', 'from_timestamp']
    ).reset_index(drop=True)

    # Write AFTER filtering so dangling.csv contains only multi-file instances.
    dangling_df.to_csv(os.path.join(CSV_OUTPUT_DIR, DANGLING_FILENAME), index=False)

    print(f"Created dangling_df with {len(dangling_df)} rows")
    return dangling_df


def _timestamp_to_relpath(ts: int) -> str:
    """Convert a unix timestamp to a relative path like YYYY-MM-DD/<ts>.csv."""
    date_str = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
    return f'{date_str}/{ts}.csv'


def _build_all_files_from_range(from_ts: int, to_ts: int) -> str:
    """Build backslash-separated file list from a timestamp range."""
    if to_ts < from_ts:
        from_ts, to_ts = to_ts, from_ts

    timestamps = list(range(from_ts, to_ts + 1, CSV_TIME_STEP_SECONDS))
    if not timestamps:
        return ''
    if timestamps[-1] != to_ts:
        timestamps.append(to_ts)

    return ALL_FILES_SEPARATOR.join(_timestamp_to_relpath(ts) for ts in timestamps)

def _build_all_files_from_timestamps(timestamps: list[int]) -> str:
    """Build backslash-separated file list from explicit timestamps (sorted, unique)."""
    if not timestamps:
        return ''
    return ALL_FILES_SEPARATOR.join(_timestamp_to_relpath(int(ts)) for ts in timestamps)


def _iter_dangling_chunks(dangling_df: pd.DataFrame, chunk_size: int):
    """Yield slices of dangling_df as DataFrames with at most chunk_size rows."""
    total = len(dangling_df)
    for start in range(0, total, chunk_size):
        yield dangling_df.iloc[start:start + chunk_size]


def create_explicit_dangling_catalog_streaming(dangling_df: pd.DataFrame, output_path: str) -> None:
    """Stream-build dangling_explicit.csv without holding all_files in memory.

    Note: this mode assumes a continuous span and expands from_timestamp..to_timestamp.
    Prefer building from dangling.buffer.csv when available (see
    create_explicit_dangling_catalog_from_buffer()).
    """
    with open(output_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['id', 'from_timestamp', 'to_timestamp', 'all_files'])

        total = len(dangling_df)
        with tqdm(total=total, desc='Explicit catalog', unit='rows') as pbar:
            for chunk in _iter_dangling_chunks(dangling_df, EXPLICIT_CHUNK_SIZE):
                _write_explicit_chunk(writer, chunk)
                pbar.update(len(chunk))


def _write_explicit_chunk(
    writer: csv.writer,
    chunk: pd.DataFrame,
) -> None:
    rows = []
    for row_id, from_ts, to_ts in zip(
        chunk['id'],
        chunk['from_timestamp'],
        chunk['to_timestamp'],
    ):
        if pd.isna(from_ts) or pd.isna(to_ts):
            all_files = ''
        else:
            all_files = _build_all_files_from_range(int(from_ts), int(to_ts))
        rows.append([row_id, from_ts, to_ts, all_files])
    writer.writerows(rows)

def create_explicit_dangling_catalog_from_buffer(buffer_path: str, output_path: str) -> None:
    """Build dangling_explicit.csv from dangling.buffer.csv (accurate file membership).

    Unlike the range-expansion approach, this writes `all_files` using the exact
    list of hourly CSV timestamps in which the id was observed, split into
    instances using the same gap rule as ``finalize_dangling_catalog``. This avoids
    incorrectly "filling in" hours between two distant observations. Single-file
    instances are included in the explicit catalog.
    """
    spanning_buffer_df = pd.read_csv(buffer_path)
    if spanning_buffer_df.empty:
        with open(output_path, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['id', 'from_timestamp', 'to_timestamp', 'all_files'])
        return

    spanning_buffer_df = spanning_buffer_df[['id', 'file']].dropna()
    spanning_buffer_df['file'] = spanning_buffer_df['file'].astype('int64')
    spanning_buffer_df = spanning_buffer_df.drop_duplicates(subset=['id', 'file'])
    spanning_buffer_df = spanning_buffer_df.sort_values(['id', 'file'], kind='mergesort').reset_index(drop=True)

    def flush_group(writer: csv.writer, row_id: str, files: list[int]) -> None:
        if row_id is None or len(files) < 1:
            return
        instance_id = f"{row_id}{INSTANCE_ID_SEPARATOR}{int(files[0])}"
        writer.writerow([instance_id, int(files[0]), int(files[-1]), _build_all_files_from_timestamps(files)])

    with open(output_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['id', 'from_timestamp', 'to_timestamp', 'all_files'])

        current_id = None
        current_files: list[int] = []
        last_ts = None

        for row_id, file_ts in spanning_buffer_df.itertuples(index=False, name=None):
            file_ts = int(file_ts)
            if row_id != current_id:
                flush_group(writer, current_id, current_files)
                current_id = row_id
                current_files = [file_ts]
                last_ts = file_ts
                continue

            if last_ts is not None and (file_ts - last_ts) > INSTANCE_GAP_SECONDS:
                flush_group(writer, current_id, current_files)
                current_files = [file_ts]
            else:
                current_files.append(file_ts)
            last_ts = file_ts

        flush_group(writer, current_id, current_files)

def remove_duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows keyed by (id, time). Returns the deduplicated DataFrame."""
    df = df.drop_duplicates(subset=['id', 'time'])
    return df


def remove_rows_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing values. Returns the filtered DataFrame."""
    df = df.dropna()
    return df

def load_dangling_csv(dangling_path: str) -> pd.DataFrame:
    """Load dangling.csv into memory, supporting headered or headerless files."""
    dangling_df = pd.read_csv(dangling_path)
    if not {'id', 'from_timestamp', 'to_timestamp'}.issubset(dangling_df.columns):
        dangling_df = pd.read_csv(dangling_path, header=None)
        dangling_df.columns = ['id', 'from_timestamp', 'to_timestamp']
    return dangling_df


def build_explicit_dangling_catalog(dangling_df: pd.DataFrame) -> None:
    """Create dangling_explicit.csv with an all_files column for each spanning flight instance.

    Streams output to avoid high memory usage.
    """
    print('Creating explicit dangling catalog... This may take a while if there are many dangling flights')
    output_path = os.path.join(CSV_OUTPUT_DIR, DANGLING_EXPLICIT_FILENAME)
    buffer_path = os.path.join(CSV_OUTPUT_DIR, DANGLING_BUFFER_FILENAME)
    if os.path.exists(buffer_path):
        create_explicit_dangling_catalog_from_buffer(buffer_path, output_path)
    else:
        create_explicit_dangling_catalog_streaming(dangling_df, output_path)
    print('Explicit dangling catalog created')

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fix CSV state files and build a catalog of dangling (spanning) flights.'
    )
    parser.add_argument(
        '--only-explicit',
        action='store_true',
        help='Skip scanning/fixing CSVs; load existing dangling.csv and build only dangling_explicit.csv.',
    )
    args = parser.parse_args()

    if args.only_explicit:
        buffer_path = os.path.join(CSV_OUTPUT_DIR, DANGLING_BUFFER_FILENAME)
        dangling_path = os.path.join(CSV_OUTPUT_DIR, DANGLING_FILENAME)
        output_path = os.path.join(CSV_OUTPUT_DIR, DANGLING_EXPLICIT_FILENAME)
        if not os.path.exists(dangling_path):
            print(f'Error: dangling CSV not found at {dangling_path}', file=sys.stderr)
            sys.exit(1)
        print(f'Loading dangling catalog from {dangling_path}')
        dangling_df = load_dangling_csv(dangling_path)
        build_explicit_dangling_catalog(dangling_df)
    else:
        fix_csv_and_create_dangling_catalog()
        dangling_df = finalize_dangling_catalog()
        build_explicit_dangling_catalog(dangling_df)


if __name__ == '__main__':
    main()
