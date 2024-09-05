import subprocess
import datetime
import pandas as pd
import requests
import os

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

jwt = get_jwt()

def execute_trino_commands(from_datetime, to_datetime):
    """
    Executes Trino shell commands for each hour within the specified datetime range.

    Args:
        from_datetime (datetime): The starting datetime.
        to_datetime (datetime): The ending datetime.
    """

    # Generate a list of Unix timestamps representing the beginning of each hour 
    hourly_timestamps = pd.date_range(from_datetime, to_datetime, freq='H').astype(int) // 10**9

    print('There are ', len(hourly_timestamps), ' splits to download')

    for timestamp in hourly_timestamps:
        # Check if the file already exists
        if os.path.exists(f"{timestamp}.csv"):
            print(f"File {timestamp}.csv already exists. Skipping download.")
            continue
        print("Current timestamp: ", timestamp)
        print("Current datetime: ", datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
        # Construct the shell command, incorporating the timestamp into the relevant parts
        command = f"./trino.jar --user=thinhhoangdinh --server=https://trino.opensky-network.org --access-token={jwt} --catalog 'minio' --schema 'osky' --execute='SELECT \
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
            AND v.time - v.lastcontact <= 15;' --output-format CSV > {timestamp}.csv" 

        # Execute the shell command
        subprocess.run(command, shell=True)  # Use shell=True for proper command parsing


from_datetime = datetime.datetime(2024, 5, 24, 0, 0, 0)  # Adjust as needed
to_datetime = datetime.datetime(2024, 6, 24, 0, 0, 0)    # Adjust as needed

execute_trino_commands(from_datetime, to_datetime)
