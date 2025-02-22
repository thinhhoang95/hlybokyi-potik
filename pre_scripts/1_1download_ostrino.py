import subprocess
import datetime
import pandas as pd
import requests
import os
from multiprocessing import Pool, cpu_count, Value, Manager

# jwt = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ0SVIwSDB0bmNEZTlKYmp4dFctWEtqZ0RYSWExNnR5eU5DWHJxUzJQNkRjIn0.eyJleHAiOjE3MjUzNTY4NjUsImlhdCI6MTcyNTM0OTY2NSwianRpIjoiMGNiYzYxZTEtNDI2Mi00MDA3LTg5MTQtZTgxN2EzNjRmM2M5IiwiaXNzIjoiaHR0cHM6Ly9hdXRoLm9wZW5za3ktbmV0d29yay5vcmcvYXV0aC9yZWFsbXMvb3BlbnNreS1uZXR3b3JrIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjEzYmYwYmQwLTMzOTktNDA2NS04ZGFiLTIyYzI0Njg1N2E4MSIsInR5cCI6IkJlYXJlciIsImF6cCI6InRyaW5vLWNsaWVudCIsInNlc3Npb25fc3RhdGUiOiIxZjQ2MzhhMy0wYjk4LTRmYzctOWNlOC1jZDBiOWJiMGI1N2UiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtb3BlbnNreS1uZXR3b3JrIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGdyb3VwcyBlbWFpbCIsInNpZCI6IjFmNDYzOGEzLTBiOTgtNGZjNy05Y2U4LWNkMGI5YmIwYjU3ZSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiVGhpbmggSG9hbmciLCJncm91cHMiOlsiL29wZW5za3kvdHJpbm8vcmVhZG9ubHkiXSwicHJlZmVycmVkX3VzZXJuYW1lIjoidGhpbmhob2FuZ2RpbmgiLCJnaXZlbl9uYW1lIjoiVGhpbmgiLCJmYW1pbHlfbmFtZSI6IkhvYW5nIiwiZW1haWwiOiJ0aGluaC5ob2FuZ2RpbmhAZW5hYy5mciJ9.GeiFz_Num5kgHA6BwosqilwlKUmdpi8fKsaPIO06PotBWLmCkrl7Y6g-os60xyILuvd31W1T-pT0-llwTPyO1PBs0VsCsOPa1mgrRyFE5uAa-QFGV_MuaLcd6BGeTR6Ss9E2vrJYcmNu630uKYu3UJOYjeCA0whkUccAUKiBiGrQohwlec0Ryz1I67rEruENt6sgV3urrywURJ8BDtJPbMnqdrG_FpMgqaWl83PEsN2aypL9Oq36fOOT68gZONgvx5s1SU6SUIDKzEVRL_V8JhBzDaY8fWJNuHaZYAsPTyUoOV_ChUSIeyvek5nm8BFsH8fjc-tUvfSXXgpZUd5jpg'

def get_jwt():
    result = requests.post(
    "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token",
    data={
        "client_id": "trino-client",
        "grant_type": "password",
        "username": "thinhhoangdinh",
        "password": "iQ6^yrwe7o3m",
        }
    )
    print('Obtained JWT: ', result.json()['access_token'][:10])
    return result.json()['access_token']

def init_worker(counter, jwt_holder):
    global downloaded_counter, shared_jwt
    downloaded_counter = counter
    shared_jwt = jwt_holder

def download_for_timestamp(timestamp):
    """
    Downloads data for a single timestamp.
    """
    global downloaded_counter, shared_jwt
    
    # Check if JWT needs to be refreshed (every 240 files)
    with downloaded_counter.get_lock():
        if downloaded_counter.value > 0 and downloaded_counter.value % 240 == 0:
            new_jwt = get_jwt()
            shared_jwt['token'] = new_jwt
    
    # Get current JWT
    current_jwt = shared_jwt['token']
    
    # Check if the file already exists
    if os.path.exists(f"summer23/raw/{timestamp}.csv"):
        with downloaded_counter.get_lock():
            downloaded_counter.value += 1
            current = downloaded_counter.value
            print(f"File {timestamp}.csv already exists. Skipping download. Progress: {current/total_files:.1%}")
        return
    
    print("Current timestamp: ", timestamp)
    print("Current datetime: ", datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
    # Get the date of the timestamp in YYYY-MM-DD format
    date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

    # Create the date folder if it doesn't exist
    os.makedirs(f'summer23/raw/{date}', exist_ok=True)

    # Check if the file already exists
    if os.path.exists(f"summer23/raw/{date}/{timestamp}.csv"):
        with downloaded_counter.get_lock():
            downloaded_counter.value += 1
            current = downloaded_counter.value
            print(f"File {timestamp}.csv already exists. Skipping download. Progress: {current/total_files:.1%}")
        return
    
    command = f"java -jar trino.jar --user=thinhhoangdinh --server=https://trino.opensky-network.org --access-token={current_jwt} --catalog 'minio' --schema 'osky' --execute='SELECT \
        v.time, v.icao24, v.lat, v.lon, v.heading, v.callsign, v.geoaltitude \
    FROM \
        state_vectors_data4 v \
    JOIN ( \
        SELECT \
            FLOOR(time / 60) AS minute, \
            MAX(time) AS recent_time \
        FROM \
            state_vectors_data4 \
        WHERE \
            hour = {timestamp} \
        GROUP BY \
            FLOOR(time / 60) \
    ) AS m \
    ON \
        FLOOR(v.time / 60) = m.minute \
        AND v.time = m.recent_time \
    WHERE \
        v.lat BETWEEN 30 AND 72 \
        AND v.lon BETWEEN -15 AND 40 \
        AND v.hour = {timestamp} \
        AND v.time - v.lastcontact <= 15;' --output-format CSV > summer23/raw/{date}/{timestamp}.csv"

    subprocess.run(command, shell=True)
    
    with downloaded_counter.get_lock():
        downloaded_counter.value += 1
        current = downloaded_counter.value
        print(f"Downloaded {timestamp}.csv. Progress: {current/total_files:.1%}")

def execute_trino_commands(from_datetime, to_datetime):
    """
    Executes Trino shell commands in parallel for each hour within the specified datetime range.
    """
    # Generate timestamps as before
    hourly_timestamps = pd.date_range(from_datetime, to_datetime, freq='H').astype('int64') // 10**9
    
    global total_files
    total_files = len(hourly_timestamps)
    print(f'There are {total_files} splits to download')
    
    # Use maximum of 4 processes or CPU count, whichever is smaller
    num_processes = min(1, cpu_count())
    print(f"Using {num_processes} processes for parallel downloads")
    
    # Create a shared counter and JWT holder
    counter = Value('i', 0)
    manager = Manager()
    jwt_holder = manager.dict()
    jwt_holder['token'] = get_jwt()  # Initial JWT
    
    # Create a pool of workers and map the download function to timestamps
    with Pool(processes=num_processes, initializer=init_worker, initargs=(counter, jwt_holder,)) as pool:
        pool.map(download_for_timestamp, hourly_timestamps)

from_datetime = datetime.datetime(2023, 4, 1, 0, 0, 0)  # Adjust as needed
to_datetime = datetime.datetime(2024, 9, 30, 0, 0, 0)    # Adjust as needed

# Create the summer23 folder
os.makedirs('summer23', exist_ok=True)

# Inside summer23 folder, create a raw folder
os.makedirs('summer23/raw', exist_ok=True)

execute_trino_commands(from_datetime, to_datetime)
