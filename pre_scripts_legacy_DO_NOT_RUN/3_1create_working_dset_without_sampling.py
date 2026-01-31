"""Create a working dataset of multi-file flights without sampling.

This script builds a working dataset from raw per-file CSVs in ``data/csv/`` by
extracting only rows that belong to flights that span multiple files (i.e.
flight *instances* listed in ``dangling.csv``). It uses ``dangling_explicit.csv``
to know which files each instance appears in, then concatenates the relevant rows
from each CSV and writes chunked output to ``data/sample/``. No down-sampling
is applied: every row belonging to a multi-file flight instance is included.

**Inputs (expected to exist):**
- ``{PATH_PREFIX}/data/csv/*.csv`` — raw flight state CSVs (time, icao24, lat,
  lon, heading, callsign, geoaltitude; no header).
- ``{PATH_PREFIX}/data/csv/dangling.csv`` — catalog of multi-file flight instances:
  columns ``id``, ``from_timestamp``, ``to_timestamp`` (header optional). ``id`` is an
  instance id: ``<callsign+icao24>|<segment_start_timestamp>``.
- ``{PATH_PREFIX}/data/csv/dangling_explicit.csv`` — explicit file list per
  flight instance: at least ``id`` and ``all_files`` (path or filename listing).

**Outputs:**
- ``{PATH_PREFIX}/data/sample/`` — directory created if missing.
- ``{PATH_PREFIX}/data/sample/{base_name}_{n}.csv`` — chunk CSVs. Each chunk
  has at most ``max_chunk_callsigns`` unique flight IDs. Columns match raw
  CSVs plus an ``id`` column (callsign + icao24).

**Parameters (main entry point):**
- ``sample_file_name`` — Base name for output chunks (e.g. ``'all.csv'``
  produces ``all_1.csv``, ``all_2.csv``, …). Extension is ignored for the
  chunk prefix.
- ``max_chunk_callsigns`` — Maximum number of unique flight IDs per output
  file; when reached, the current dataframe is written and a new chunk is
  started.

**Example (CLI):**
  Run from project root (or ensure ``path_prefix`` and imports resolve)::

      python pre_scripts/3_1create_working_dset_without_sampling.py

  This calls ``sample_working_dataset('all.csv')``, producing e.g.
  ``data/sample/all_1.csv``, ``data/sample/all_2.csv``, … .

**Example (programmatic):**
  After running this script or importing ``sample_working_dataset``::

  >>> sample_working_dataset('working.csv', max_chunk_callsigns=5000)
  >>> # Output: data/sample/working_1.csv, data/sample/working_2.csv, ...
"""
import sys, os
import time 
import pandas as pd
from tqdm import tqdm
# Add the path prefix to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from path_prefix import PATH_PREFIX
from data_preambles import csv_to_exclude, dtypes_no_id, catalog_col_names, col_names

# Instance id separator used by dangling.csv/dangling_explicit.csv:
# instance_id = "<callsign+icao24>|<segment_start_timestamp>"
INSTANCE_ID_SEPARATOR = '|'

def create_dir_if_not_exists() -> None:
    """Create the data/sample directory under PATH_PREFIX if it does not exist."""
    if not os.path.exists(f'{PATH_PREFIX}/data/sample'):
        os.makedirs(f'{PATH_PREFIX}/data/sample')

# Too slow, we don't use this
# ====================================
# def get_csv_files_from_to(from_file: str, to_file: str, file_catalog: pd.DataFrame) -> list[str]:
#     # get the index of from_file and to_file in file_catalog
#     from_index = file_catalog.index[file_catalog['file'] == from_file]
#     to_index = file_catalog.index[file_catalog['file'] == to_file]
#     # Convert from_index and to_index to int
#     from_index = from_index[-1]
#     to_index = to_index[-1]
#     # concatenate all the strings in results
#     return '/'.join(file_catalog.iloc[from_index:to_index+1]['file'].tolist())

def get_instance_mapping_in_file(explicit_catalog: pd.DataFrame, file: str, instance_ids: list) -> dict:
    """Return mapping base_id -> instance_id for instances that include `file`.

    Used to select which multi-file flight IDs actually appear in a given
    CSV file so only relevant rows are admitted.

    Parameters
    ----------
    explicit_catalog : pd.DataFrame
        Catalog with columns 'id' and 'all_files' (e.g. dangling_explicit.csv).
    file : str
        CSV filename (or path segment) to match in 'all_files'.
    instance_ids : list
        Candidate instance IDs to filter.

    Returns
    -------
    dict
        Mapping {base_id: instance_id} for instances whose 'all_files' contains `file`.
    """
    catalog_filtered = explicit_catalog[explicit_catalog['id'].isin(instance_ids)]
    file_mask = catalog_filtered['all_files'].str.contains(file, regex=False)
    subset = catalog_filtered[file_mask]
    return dict(zip(subset['base_id'], subset['id']))

def sample_working_dataset(sample_file_name: str = 'sample.csv', max_chunk_callsigns: int = 10000) -> None:
    """Build the working dataset of multi-file flights and write chunked CSVs to data/sample.

    Reads all raw CSVs in data/csv (excluding catalog/metadata files), the
    dangling catalog, and dangling_explicit; for each CSV, keeps only rows
    whose flight id is in the multi-file catalog and appears in that file.
    Accumulates rows and writes a new chunk whenever the number of unique
    flight IDs reaches max_chunk_callsigns.

    Parameters
    ----------
    sample_file_name : str, default 'sample.csv'
        Base name for output files; extension is dropped for chunk names
        (e.g. 'all.csv' -> all_1.csv, all_2.csv).
    max_chunk_callsigns : int, default 10000
        Maximum unique flight IDs per output chunk; triggers a new file when
        reached (and at the end for the remainder).
    """
    create_dir_if_not_exists()
    # List all csv files in the data/csv directory
    csv_files = [file for file in os.listdir(f'{PATH_PREFIX}/data/csv') if file.endswith('.csv')]
    # Remove dangling.buffer.csv and dangling.csv and catalog.csv if they exist
    csv_files = [file for file in csv_files if file not in csv_to_exclude]
    csv_files.sort()
    print(f'Found {len(csv_files)} csv files')

    # Read the dangling catalog (supports with or without header).
    catalog_path = f'{PATH_PREFIX}/data/csv/dangling.csv'
    catalog = pd.read_csv(catalog_path, dtype=dtypes_no_id)
    if not set(catalog_col_names).issubset(catalog.columns):
        catalog = pd.read_csv(catalog_path, dtype=dtypes_no_id, header=None)
        catalog.columns = catalog_col_names

    df_sample = pd.DataFrame()
    chunk_number = 1

    def save_chunk(df, chunk_num):
        """Write the current dataframe to data/sample/{base}_{chunk_num}.csv and return an empty DataFrame."""
        chunk_name = f'{sample_file_name.split(".")[0]}_{chunk_num}.csv'
        df.to_csv(f'{PATH_PREFIX}/data/sample/{chunk_name}', index=False)
        print(f'Saved chunk {chunk_num} with {len(df)} rows and {df["id"].nunique()} unique ids')
        return pd.DataFrame()

    # Process one-file callsigns: we don't use this because the current dataset does not contain one-file callsigns
    # print('Processing one-file callsigns')
    # for file in csv_files:
    #     df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{file}')
    #     print(f'Processing file {file}')
    #     list_id_in_df = df['id'].unique()
    #     id_one_file = list_id_in_df[~np.isin(list_id_in_df, catalog['id'])]

    #     df_one_file = df[df['id'].isin(id_one_file)]
    #     print(f'{len(df_one_file)} rows found for one-file callsigns')
        
    #     df_sample = pd.concat([df_sample, df_one_file])
        
    #     if df_sample['id'].nunique() >= max_chunk_callsigns:
    #         df_sample = save_chunk(df_sample, chunk_number)
    #         chunk_number += 1

    # Process multi-file callsigns
    print('Processing multi-file callsigns')
    instance_ids_multi_file = catalog['id'].unique()
    print('Some instance ids in catalog: ', instance_ids_multi_file[:10])

    print('Reading dangling_explicit.csv...')
    explicit_catalog = pd.read_csv(f'{PATH_PREFIX}/data/csv/dangling_explicit.csv')
    # Derive base_id from instance_id for matching to raw CSV rows
    explicit_catalog['base_id'] = explicit_catalog['id'].str.rsplit(INSTANCE_ID_SEPARATOR, n=1).str[0]

    for file in tqdm(csv_files):
        df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{file}', dtype=dtypes_no_id, header=None)
        df.columns = col_names
        # Create the id column for df   
        df['id'] = df['callsign'] + df['icao24']
        
        # For each file, and for each flight spanning over multiple files, we get
        # all the IDs that present in this file, so we can concatenate all the
        # relevant rows here
        instance_map = get_instance_mapping_in_file(explicit_catalog, file, instance_ids_multi_file)
        if not instance_map:
            continue
        df_admit = df[df['id'].isin(instance_map.keys())].copy()
        # Replace base id with instance id so downstream processing uses per-flight instances
        df_admit['id'] = df_admit['id'].map(instance_map)
        df_sample = pd.concat([df_sample, df_admit])

        if df_sample['id'].nunique() >= max_chunk_callsigns:
            df_sample = save_chunk(df_sample, chunk_number)
            chunk_number += 1

    # Save any remaining data
    if not df_sample.empty:
        save_chunk(df_sample, chunk_number)


if __name__ == '__main__':
    sample_working_dataset('all.csv')
