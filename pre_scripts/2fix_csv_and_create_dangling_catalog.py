import sys, os 
import pandas as pd
import csv
# Add the path prefix to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from path_prefix import PATH_PREFIX

from data_preambles import dtypes_no_id, csv_to_exclude, col_names


def fix_csv_and_create_dangling_catalog() -> None:
    # # If catalog.csv exists, read it
    # if os.path.exists(f'{PATH_PREFIX}/data/csv/catalog.csv'):
    #     spanning_df = pd.read_csv(f'{PATH_PREFIX}/data/csv/catalog.csv')
    # else:
    spanning_buffer_file = f'{PATH_PREFIX}/data/csv/dangling.buffer.csv'
    # Create a CSV writer for the spanning_buffer_file
    spanning_buffer_writer = csv.writer(open(spanning_buffer_file, 'w'))
    spanning_buffer_writer.writerow(['id', 'file'])

    # List all csv files in the data/csv directory
    csv_files = [file for file in os.listdir(f'{PATH_PREFIX}/data/csv') if file.endswith('.csv')]
    # Remove catalog.csv from csv_files if it exists
    csv_files = [file for file in csv_files if file not in csv_to_exclude]

    print(f'Found {len(csv_files)} csv files in the data/csv directory')

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
        
        df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{csv_file}', header=None, dtype=dtypes_no_id)
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
        timestamp_of_csv_file = int(csv_file.replace('.csv', ''))
        spanning_buffer_writer.writerows([[id, timestamp_of_csv_file] for id in this_csv_ids])


        # Save the fixed CSV file (not related to dangling flights, just some fixing of format and adding a header row)
        # df.to_csv(f'{PATH_PREFIX}/data/csv/{csv_file}', index=False)


def finalize_dangling_catalog() -> None:
    print('Finalizing the dangling catalog... This may take a while')
    dangling_df = pd.DataFrame(columns=['id', 'from_timestamp', 'to_timestamp'])
    # Finally, we perform an SQL-like operation on the spanning_buffer_file: we group by id, set from_timestamp to the minimum timestamp and to_timestamp to the maximum timestamp
    # Read the spanning_buffer_file
    spanning_buffer_df = pd.read_csv(f'{PATH_PREFIX}/data/csv/dangling.buffer.csv')
    
    # Group by id, get min and max timestamps
    dangling_df = spanning_buffer_df.groupby('id').agg({
        'file': ['min', 'max']
    }).reset_index()
    
    # Flatten column names and rename
    dangling_df.columns = ['id', 'from_timestamp', 'to_timestamp']
    
    # Sort the dataframe by id
    dangling_df = dangling_df.sort_values('id')
    
    # Reset the index
    dangling_df = dangling_df.reset_index(drop=True)
    
    # Save the dangling_df to a csv file
    dangling_df.to_csv(f'{PATH_PREFIX}/data/csv/dangling.csv', index=False)

    # Remove all rows with from_timestamp >= to_timestamp
    dangling_df = dangling_df[dangling_df['from_timestamp'] < dangling_df['to_timestamp']]
    
    print(f"Created dangling_df with {len(dangling_df)} rows")
    return dangling_df

def create_explicit_dangling_catalog(dangling_df: pd.DataFrame) -> pd.DataFrame:
    all_csv_files = [file for file in os.listdir(f'{PATH_PREFIX}/data/csv') if file.endswith('.csv')]
    all_csv_files.sort()
    all_csv_files = [file for file in all_csv_files if file not in csv_to_exclude]

    print(f'There are {len(all_csv_files)} csv files in the data/csv directory')
    # Create a dataframe for the files
    file_catalog = pd.DataFrame({'file': all_csv_files})
    file_catalog['timestamp'] = file_catalog['file'].str.replace('.csv', '').astype(int)
    file_catalog = file_catalog.sort_values('timestamp')

    # Clone the dangling_df
    dangling_df_explicit = dangling_df.copy()

    # Create a dictionary mapping timestamps to file indices
    timestamp_to_index = dict(zip(file_catalog['timestamp'], file_catalog.index))

    # Vectorized operation to get start and end indices
    start_indices = dangling_df_explicit['from_timestamp'].map(timestamp_to_index)
    end_indices = dangling_df_explicit['to_timestamp'].map(timestamp_to_index)

    # Create the all_files column
    dangling_df_explicit['all_files'] = [
        '/'.join(file_catalog['file'].iloc[start:end+1])
        for start, end in tqdm(zip(start_indices, end_indices))
    ]

    return dangling_df_explicit
    


def remove_duplicated_rows(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(subset=['id', 'time'])
    return df

def remove_rows_with_missing_values(df: pd.DataFrame) -> None:
    df = df.dropna()
    return df

def build_explicit_dangling_catalog(dangling_df: pd.DataFrame) -> pd.DataFrame:
    # Create dangling_explicit.csv to write out the explicit dangling catalog (i.e., all_files column show all the files that the flight spans)
    print(f'Creating explicit dangling catalog... This may take a while if there are many dangling flights')
    dangling_df_explicit = create_explicit_dangling_catalog(dangling_df)
    dangling_df_explicit.to_csv(f'{PATH_PREFIX}/data/csv/dangling_explicit.csv', index=False)
    print(f'Explicit dangling catalog created')

if __name__ == '__main__':
    fix_csv_and_create_dangling_catalog()
    dangling_df = finalize_dangling_catalog()
    build_explicit_dangling_catalog(dangling_df)