import sys, os
import time 
import pandas as pd
from tqdm import tqdm
# Add the path prefix to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from path_prefix import PATH_PREFIX
from data_preambles import csv_to_exclude, dtypes_no_id, catalog_col_names, col_names

def create_dir_if_not_exists() -> None:
    # Create the data/sample directory if it doesn't exist
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

def get_ids_in_file(explicit_catalog: pd.DataFrame, file: str, ids: list) -> list: # catalog is dangling.csv
    catalog_filtered = explicit_catalog[explicit_catalog['id'].isin(ids)]
    # return the ids that file is in all_files
    return catalog_filtered[catalog_filtered['all_files'].str.contains(file)]['id'].unique()

def sample_working_dataset(sample_file_name: str = 'sample.csv', max_chunk_callsigns: int = 10000) -> None:
    create_dir_if_not_exists()
    # List all csv.gz files in the data/csv directory
    csv_files = [file for file in os.listdir(f'{PATH_PREFIX}/data/csv') if file.endswith('.csv')]
    # Remove dangling.buffer.csv and dangling.csv and catalog.csv if they exist
    csv_files = [file for file in csv_files if file not in csv_to_exclude]
    csv_files.sort()
    print(f'Found {len(csv_files)} csv files')

    # Read the catalog.csv file
    catalog = pd.read_csv(f'{PATH_PREFIX}/data/csv/dangling.csv', dtype=dtypes_no_id, header=None)
    # Add column names to catalog DF
    catalog.columns = catalog_col_names

    df_sample = pd.DataFrame()
    chunk_number = 1

    def save_chunk(df, chunk_num):
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
    id_multi_file = catalog['id'].unique()
    print('Some ids in id_multi_file: ', id_multi_file[:10])

    print('Reading dangling_explicit.csv...')
    explicit_catalog = pd.read_csv(f'{PATH_PREFIX}/data/csv/dangling_explicit.csv')

    for file in tqdm(csv_files):
        df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{file}', dtype=dtypes_no_id, header=None)
        df.columns = col_names
        # Create the id column for df   
        df['id'] = df['callsign'] + df['icao24']
        
        ids_to_process = get_ids_in_file(explicit_catalog, file, id_multi_file)
        df_admit = df[df['id'].isin(ids_to_process)]
        df_sample = pd.concat([df_sample, df_admit])

        if df_sample['id'].nunique() >= max_chunk_callsigns:
            df_sample = save_chunk(df_sample, chunk_number)
            chunk_number += 1

    # Save any remaining data
    if not df_sample.empty:
        save_chunk(df_sample, chunk_number)


if __name__ == '__main__':
    sample_working_dataset('all.csv')