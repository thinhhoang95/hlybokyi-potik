import sys, os 
import pandas as pd
# Add the path prefix to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_prefix import PATH_PREFIX

def fix_csv_and_create_dangling_catalog() -> None:
    # If catalog.csv exists, read it
    if os.path.exists(f'{PATH_PREFIX}/data/csv/catalog.csv'):
        spanning_df = pd.read_csv(f'{PATH_PREFIX}/data/csv/catalog.csv')
    else:
        spanning_df = pd.DataFrame(columns=['id', 'from', 'to'])


    # List all csv.gz files in the data/csv directory
    csv_files = [file for file in os.listdir(f'{PATH_PREFIX}/data/csv') if file.endswith('.csv.gz')]
    print(f'Found {len(csv_files)} csv.gz files')

    # Sort the csv_files list
    csv_files.sort()

    # For each csv.gz file, read the csv.gz file and add an id column
    for file_id, csv_file in enumerate(csv_files):
        print(f'Processing {csv_file}')
        df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{csv_file}')
        df['id'] = df['callsign'] + df['icao24']
        # only keep the following columns: id, lat, lon, velocity, heading, vertrate, baroaltitude, geoaltitude, lastposupdate
        df = df[['id', 'lat', 'lon', 'velocity', 'heading', 'vertrate', 'baroaltitude', 'geoaltitude', 'lastposupdate']]
        # remove duplicated rows
        df = remove_duplicated_rows(df)
        # remove rows with missing values
        df = remove_rows_with_missing_values(df)
        print('CSV fixes applied. Now handling spanning flights...')

        # Handle spanning flights
        this_csv_ids = df['id'].unique()
        catalog_ids = spanning_df['id'].unique()
        # Find the intersection of this_csv_ids and catalog_ids
        intersection = set(this_csv_ids) & set(catalog_ids)
        new_catalog_ids = list(set(this_csv_ids) | intersection)
        # Remove from spanning_df all the rows with id not in new_catalog_ids
        spanning_df = spanning_df[spanning_df['id'].isin(new_catalog_ids)]
        # Append ids that are in this_csv_ids but not in intersection to spanning_df
        new_ids = set(this_csv_ids) - intersection
        new_rows = pd.DataFrame({
            'id': list(new_ids),
            'from': csv_file,
            'to': csv_file
        })
        spanning_df = pd.concat([spanning_df, new_rows], ignore_index=True)
        # Set the 'to' column of spanning_df to csv_file
        spanning_df['to'] = csv_file

        df.to_csv(f'{PATH_PREFIX}/data/csv/{csv_file}', index=False, compression='gzip')

    # Finally, remove all rows with 'from' equals to 'to' of spanning_df
    spanning_df = spanning_df[spanning_df['from'] != spanning_df['to']]
    # Save spanning_df to a CSV file
    spanning_df.to_csv(f'{PATH_PREFIX}/data/csv/dangling.csv', index=False)

def remove_duplicated_rows(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(subset=['id', 'lastposupdate'])
    return df

def remove_rows_with_missing_values(df: pd.DataFrame) -> None:
    df = df.dropna()
    return df

if __name__ == '__main__':
    fix_csv_and_create_dangling_catalog()
