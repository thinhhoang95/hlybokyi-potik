import requests
import json
import os
from pathlib import Path

def download_route(icao_from: str, icao_to: str):
    # API configuration
    api_key = "prcwCuQM2XxIeMwCFqIye2QZi3s6MkHlO4E3dqXA"
    base_url = "https://api.flightplandatabase.com/search/plans"
    
    # Prepare request parameters
    params = {
        "fromICAO": icao_from,
        "toICAO": icao_to,
        "limit": 1,
        "includeRoute": 'true'
    }
    
    # Make API request with basic auth
    response = requests.get(
        base_url,
        params=params,
        auth=(api_key, "")
    )
    
    # Ensure the routes directory exists
    routes_dir = Path("flight_schedules/routes")
    routes_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename and save response
    filename = f"{icao_from}-{icao_to}.json"
    filepath = routes_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(response.json(), f, indent=2)

import pandas as pd
from tqdm import tqdm
import time
if __name__ == "__main__":
    # Create directory for routes if not exists
    routes_dir = Path("flight_schedules/routes")
    routes_dir.mkdir(parents=True, exist_ok=True)

    # Load OD pairs
    df_od_pairs = pd.read_csv("flight_schedules/processed_data/od_pairs.csv")

    od_pairs = df_od_pairs['OD'].tolist()
    for od_pair in tqdm(od_pairs[1200:2400]):
        try:
            # Check if the route file already exists
            icao_from, icao_to = od_pair.split('-')
            filename = f"{icao_from}-{icao_to}.json"
            filepath = routes_dir / filename
            if filepath.exists():
                print(f"Route {od_pair} already exists")
                continue
            download_route(icao_from, icao_to)
            # Pause for 5 seconds before downloading the next route to avoid API abuse
            time.sleep(5)
        except Exception as e:
            print(f"Error downloading route {od_pair}: {e}")
            continue

