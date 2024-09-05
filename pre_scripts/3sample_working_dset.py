import sys, os 
import pandas as pd
# Add the path prefix to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from path_prefix import PATH_PREFIX

N_CALLSIGNS = 1_000 # number of callsigns for one-file callsigns
N_CALLSIGNS_MULTI = 1_000 # number of callsigns for multi-file callsigns

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

def sample_working_dataset(n_callsigns: int = N_CALLSIGNS, sample_file_name: str = 'sample.csv', n_callsigns_multi: int = N_CALLSIGNS_MULTI) -> None:
    create_dir_if_not_exists()
    # List all csv.gz files in the data/csv directory
    csv_files = [file for file in os.listdir(f'{PATH_PREFIX}/data/csv') if file.endswith('.csv.gz')]
    csv_files.sort()
    print(f'Found {len(csv_files)} csv.gz files')

    n_files = len(csv_files)
    n_callsigns_per_file = n_callsigns // n_files
    
    # Read the catalog.csv file
    catalog = pd.read_csv(f'{PATH_PREFIX}/data/csv/catalog.csv')

    df_sample = pd.DataFrame()

    # Sampling loop for one_file callsigns
    print('Sampling one-file callsigns')
    for file in csv_files:
        df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{file}')
        print(f'Processing file {file}')
        list_id_in_df = df['id'].unique()
        list_id_in_df = np.random.choice(list_id_in_df, n_callsigns_per_file, replace=False)
        # We differentiate between "one-file" callsigns and "multi-file" callsigns
        # id_one_file contains ids in df_id that are not found in catalog['id']
        id_one_file = list_id_in_df[~np.isin(list_id_in_df, catalog['id'])]
        # id_multi_file contains ids in df_id that are found in catalog['id']
        id_multi_file = list_id_in_df[np.isin(list_id_in_df, catalog['id'])]

        # For all ids in id_one_file, we collect every row in df having that id
        df_one_file = df[df['id'].isin(id_one_file)]
        print(f'{len(df_one_file)} rows admitted for one-file callsigns')
        df_sample = pd.concat([df_sample, df_one_file])
    
    # Sampling loop for multi-file callsigns
    print('Sampling multi-file callsigns')
    # id_multi_file samples from the id column of the catalog
    id_multi_file = catalog['id'].unique()
    id_multi_file = np.random.choice(id_multi_file, n_callsigns_multi, replace=False)
    print('Some ids in id_multi_file: ', id_multi_file[:10])
    for file in csv_files:
        df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{file}')
        print(f'Processing file {file}')

        ids_to_process = get_ids_in_file(catalog, file, id_multi_file)
        print(f'{len(ids_to_process)} ids to process for file {file}')
        df_admit = df[df['id'].isin(ids_to_process)]
        df_sample = pd.concat([df_sample, df_admit])
        print(f'{len(df_admit)} rows admitted for multi-file callsigns')
            
    # Sort df_sample by lastposupdate 
    df_sample = df_sample.sort_values(by='lastposupdate', ascending=True)
    # Write the sample to a csv file
    df_sample.to_csv(f'{PATH_PREFIX}/data/sample/{sample_file_name}', index=False)

if __name__ == '__main__':
    sample_working_dataset(N_CALLSIGNS, 'sample.csv', N_CALLSIGNS_MULTI)