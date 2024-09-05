import sys, os 
import pandas as pd
# Add the path prefix to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from path_prefix import PATH_PREFIX

def create_dir_if_not_exists() -> None:
    # Create the data/sample directory if it doesn't exist
    if not os.path.exists(f'{PATH_PREFIX}/data/sample'):
        os.makedirs(f'{PATH_PREFIX}/data/sample')

def get_csv_files_from_to(from_file: str, to_file: str) -> list[str]:
    # List all csv.gz files in the data/csv directory
    csv_files = [file for file in os.listdir(f'{PATH_PREFIX}/data/csv') if file.endswith('.csv.gz')]
    csv_files.sort()
    results = csv_files[csv_files.index(from_file):csv_files.index(to_file)+1]
    # concatenate all the strings in results
    return '/'.join(results)

def get_ids_in_file(catalog: pd.DataFrame, file: str, ids: list) -> list:
    catalog_filtered = catalog[catalog['id'].isin(ids)]
    catalog_filtered['all_files'] = catalog_filtered.apply(lambda row: get_csv_files_from_to(row['from'], row['to']), axis=1)
    # return the ids that file is in all_files
    return catalog_filtered[catalog_filtered['all_files'].str.contains(file)]['id'].unique()

def sample_working_dataset(sample_file_name: str = 'sample.csv', max_chunk_callsigns: int = 10000) -> None:
    create_dir_if_not_exists()
    # List all csv.gz files in the data/csv directory
    csv_files = [file for file in os.listdir(f'{PATH_PREFIX}/data/csv') if file.endswith('.csv.gz')]
    csv_files.sort()
    print(f'Found {len(csv_files)} csv.gz files')

    # Read the catalog.csv file
    catalog = pd.read_csv(f'{PATH_PREFIX}/data/csv/catalog.csv')

    df_sample = pd.DataFrame()
    chunk_number = 1

    def save_chunk(df, chunk_num):
        chunk_name = f'{sample_file_name.split(".")[0]}_{chunk_num}.csv'
        df.to_csv(f'{PATH_PREFIX}/data/sample/{chunk_name}', index=False)
        print(f'Saved chunk {chunk_num} with {len(df)} rows and {df["id"].nunique()} unique ids')
        return pd.DataFrame()

    # Process one-file callsigns
    print('Processing one-file callsigns')
    for file in csv_files:
        df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{file}')
        print(f'Processing file {file}')
        list_id_in_df = df['id'].unique()
        id_one_file = list_id_in_df[~np.isin(list_id_in_df, catalog['id'])]

        df_one_file = df[df['id'].isin(id_one_file)]
        print(f'{len(df_one_file)} rows found for one-file callsigns')
        
        df_sample = pd.concat([df_sample, df_one_file])
        
        if df_sample['id'].nunique() >= max_chunk_callsigns:
            df_sample = save_chunk(df_sample, chunk_number)
            chunk_number += 1

    # Process multi-file callsigns
    print('Processing multi-file callsigns')
    id_multi_file = catalog['id'].unique()
    print('Some ids in id_multi_file: ', id_multi_file[:10])
    for file in csv_files:
        df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{file}')
        print(f'Processing file {file}')

        ids_to_process = get_ids_in_file(catalog, file, id_multi_file)
        print(f'{len(ids_to_process)} ids to process for file {file}')
        df_admit = df[df['id'].isin(ids_to_process)]
        df_sample = pd.concat([df_sample, df_admit])
        print(f'{len(df_admit)} rows found for multi-file callsigns')

        if df_sample['id'].nunique() >= max_chunk_callsigns:
            df_sample = save_chunk(df_sample, chunk_number)
            chunk_number += 1

    # Save any remaining data
    if not df_sample.empty:
        save_chunk(df_sample, chunk_number)


if __name__ == '__main__':
    sample_working_dataset('all.csv')