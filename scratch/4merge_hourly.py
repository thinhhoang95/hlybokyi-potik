import os
import pandas as pd
import yaml
from collections import defaultdict
from tqdm import tqdm

def load_merge_instructions(yaml_path, max_files_per_id=5):
    """
    Load the merge instructions from a YAML file and limit each ID to the first `max_files_per_id` files.
    """
    print(f"Reading instructions from {yaml_path}...")
    with open(yaml_path, 'r') as file:
        instructions = yaml.safe_load(file)
    
    # Limit each ID to the first `max_files_per_id` entries
    limited_instructions = {id_: files[:max_files_per_id] for id_, files in instructions.items()}
    return limited_instructions

def group_ids_by_file(merge_instructions):
    """
    Create a mapping from file timestamp to list of IDs that have this file in their instruction list.
    """
    file_to_ids = defaultdict(list)
    for id_, files in merge_instructions.items():
        for file in files:
            file_to_ids[file].append(id_)
    return file_to_ids


def process_batch(batch_files, merge_instructions, file_to_ids, data_dir,
                  output_dir, batch_id, batch_size):
    """
    Process a batch of files:
    - Load each file into a DataFrame.
    - For each ID that needs to have its rows moved, move them to the first file.
    - Remove moved rows from other files.
    - Save the updated DataFrames back to CSV.
    """

    # Load all DataFrames in the batch
    dfs = {}
    # Convert batch_files to a list of ints
    batch_files = [int(file) for file in batch_files]
    for file in batch_files:
        file_path = os.path.join(data_dir, f"{file}.csv")
        if os.path.exists(file_path):
            dfs[file] = pd.read_csv(file_path, dtype={'id': str})
        else:
            print(f"Warning: {file_path} does not exist.")
            dfs[file] = pd.DataFrame()  # Empty DataFrame

    # We use a set to avoid duplicate processing
    processed_ids = set()

    for file in batch_files:
        ids = file_to_ids.get(file, []) # ids that will be merged
        for id_ in ids:
            if id_ in processed_ids: # We are meeting the processed ID
                continue 
                # # Get the index of the processed ID
                # idx = processed_ids.index(id_)
                # # We remove all rows with ID from all files in the batch where the timestamp is larger
                # # than the first file of the processed ID
                # for df_timestamp, df in dfs.items():
                #     if df_timestamp > first_files_of_processed_ids[idx] and not df.empty:
                #         # Remove rows where ID matches any ID in processed_ids (case insensitive)
                #         dfs[df_timestamp] = df[~df['id'].str.strip().str.lower().isin(
                #             [pid.strip().lower() for pid in processed_ids])]
                # # Then we skip this ID
                # continue 

            file_list = merge_instructions.get(id_, [])
            if not file_list:
                continue  # No instructions for this ID

            first_file = file_list[0]
            if first_file not in dfs:
                # The first file might not be in the current batch
                continue
            
            first_df = dfs[first_file]
            # Skip if ID not found in first file
            if not first_df[first_df['id'].str.strip().str.lower() == id_.strip().lower()].shape[0]:
                continue
            

            # Iterate through all files for this ID
            for idx, file_timestamp in enumerate(file_list):
                if file_timestamp not in dfs:
                    continue  # File not in current batch

                df = dfs[file_timestamp]
                if df.empty:
                    continue  # Nothing to do

                if file_timestamp == first_file:
                    # Ensure that the first file contains all rows for this ID
                    # If already present, do nothing
                    pass
                else:
                    # Extract rows for this ID and append to the first file
                    id_rows = df[df['id'].str.strip().str.lower() == id_.strip().lower()]
                    if not id_rows.empty:
                        first_df = pd.concat([first_df, id_rows], ignore_index=True)
                        dfs[first_file] = first_df

                        # Remove the rows from the current file
                        dfs[file_timestamp] = df[df['id'].str.strip().str.lower() != id_.strip().lower()]

            processed_ids.add(id_)

    # Save all DataFrames back to their respective CSV files
    for file, df in dfs.items():
        file_path = os.path.join(output_dir, f"{file}.csv")
        df.to_csv(file_path, index=False)

    pass


def merge_csv_files(data_dir, yaml_path, output_dir, batch_size=5):
    """
    Main function to merge CSV files based on merge instructions.
    """
    # Load and limit merge instructions
    merge_instructions = load_merge_instructions(yaml_path, max_files_per_id=5)

    # Create a mapping from file to IDs
    file_to_ids = group_ids_by_file(merge_instructions)

    import glob

    # Get all CSV files from data directory, excluding hidden files
    all_files = [os.path.splitext(os.path.basename(f))[0] 
                 for f in glob.glob(os.path.join(data_dir, "*.csv"))
                 if not os.path.basename(f).startswith("._")]
    
    # Sort the files in ascending order
    all_files.sort()

    # all_files = ['1716901200','1716904800','1716908400','1716912000','1716915600']

    # Get all unique file timestamps
    # all_files = sorted(file_to_ids.keys())

    # Process files in batches
    for i in tqdm(range(0, len(all_files), 1), desc="Processing Batches"):
        batch_files = all_files[i:i + batch_size]
        process_batch(batch_files, merge_instructions, file_to_ids, data_dir, output_dir,
                      batch_id=i, batch_size=batch_size)

    print("Merging completed successfully.")

from dotenv import load_dotenv

load_dotenv()
print(f'PROJECT_ROOT: {os.getenv("PROJECT_ROOT")}')
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

if __name__ == "__main__":
    # Define paths
    DATA_DIR = f"{PROJECT_ROOT}/data/hourly_merged"  # Directory where CSV files are stored
    YAML_PATH = f"{PROJECT_ROOT}/data/catalog/merge_instructions.yml"
    OUTPUT_DIR = DATA_DIR

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ensure the data directory exists
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"The data directory '{DATA_DIR}' does not exist.")

    # Run the merge process
    merge_csv_files(DATA_DIR, YAML_PATH, OUTPUT_DIR, batch_size=5)
