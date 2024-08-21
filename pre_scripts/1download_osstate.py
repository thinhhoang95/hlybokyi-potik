from tqdm import tqdm
import requests
from datetime import datetime, timedelta
from typing import List
import multiprocessing

import os, sys
# add parent directory to sys.path to import path_prefix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_prefix import PATH_PREFIX

def make_dir() -> None:
    # Create the tar and the csv directories under data
    os.makedirs(f'{PATH_PREFIX}/data/tar', exist_ok=True)
    os.makedirs(f'{PATH_PREFIX}/data/csv', exist_ok=True)

make_dir()

def download_file(url: str, filename: str, directory: str) -> None:
    print(f"Downloading {filename} from {url} to {directory}")
    response = requests.get(url)
    with open('{}/{}'.format(directory, filename), 'wb') as file:
        file.write(response.content)

def download(datetimes: List) -> None:
    for datetime in tqdm(datetimes):
        hours = range(24) # 0-23
        hours = ['00', '01', '02'] # for testing purpose
        hours = [str(hour).zfill(2) for hour in hours] # 00-23

        for hour in hours:
            try:
                url = 'https://opensky-network.org/datasets/states/{datetime}/{hour}/states_{datetime}-{hour}.csv.tar'
                
                url = url.format(datetime=datetime, hour=hour)
                print(f'Downloading from {url}')
                filename = url.split('/')[-1]
                download_file(url, filename, f'{PATH_PREFIX}/data/tar')
            except Exception as e:
                print(f"Error occurred while downloading file: {e}")
                continue

def get_date_list(std: str, ed: str) -> List[str]:
    start_date = datetime.strptime(std, '%Y-%m-%d')
    end_date = datetime.strptime(ed, '%Y-%m-%d')
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=7)
    return date_list

# def test_get_date_list() -> None:
#     date_list = get_date_list('2022-04-04', '2022-06-27')
#     assert date_list[0] == '2022-04-04'
#     assert date_list[-1] == '2022-06-27'
#     assert len(date_list) == 26

# test_get_date_list()

# def test_download_file() -> None:
#     return # this function should be tested separately
#     url = 'https://opensky-network.org/datasets/states/2022-06-06/00/states_2022-06-06-00.csv.tar'
#     filename = 'states_2022-06-06-00.csv.tar'
#     directory = '/workspace/deepflow/data/osstate'
#     download_file(url, filename, directory)

def extract_tar() -> None:
    # List all the tar files in the tar directory
    tar_files = os.listdir(f'{PATH_PREFIX}/data/tar')
    for tar_file in tar_files:
        # Extract the tar file
        os.system(f'tar -xvf {PATH_PREFIX}/data/tar/{tar_file} -C {PATH_PREFIX}/data/csv')
    print(f'Extracted {len(tar_files)} tar files')

def rename_csv() -> None:
    csv_files = os.listdir(f'{PATH_PREFIX}/data/csv')
    csv_files = [file for file in csv_files if file.endswith('.csv.gz')]
    for csv_file in csv_files:
        # Remove the states_ prefix
        new_name = csv_file.replace('states_', '')
        os.rename(f'{PATH_PREFIX}/data/csv/{csv_file}', f'{PATH_PREFIX}/data/csv/{new_name}')
    print(f'Renamed {len(csv_files)} csv files')

def delete_non_gz() -> None:
    csv_files = os.listdir(f'{PATH_PREFIX}/data/csv')
    csv_files = [file for file in csv_files if not file.endswith('.csv.gz')]
    for csv_file in csv_files:
        os.remove(f'{PATH_PREFIX}/data/csv/{csv_file}')
    print(f'Deleted {len(csv_files)} non-gz files')

def multiprocess_download(threads=4) -> None:
    date_list = get_date_list('2022-05-23', '2022-05-23')
    sublist_size = len(date_list) // threads
    print(f"Sublist size: {sublist_size}")
    print(f"Date list size: {len(date_list)}")

    processes = []
    for i in range(threads):
        start_index = i * sublist_size
        end_index = start_index + sublist_size if i < threads - 1 else len(date_list)
        sublist = date_list[start_index:end_index]

        process = multiprocessing.Process(target=download, args=(sublist,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    # multiprocess_download()
    extract_tar()
    rename_csv()
    delete_non_gz()