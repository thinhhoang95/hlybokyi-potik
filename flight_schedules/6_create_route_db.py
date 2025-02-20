import pandas as pd
import json

def process_json_file_for_route(filepath: str):
    # Load the JSON file
    with open(filepath, 'r') as f:
        data = json.load(f)

    # If the dataframe is empty, return an empty dataframe
    if data is None or len(data) == 0:
        return pd.DataFrame()
    
    # Extract the route nodes
    route_nodes = data[0]['route']['nodes']
    
    # Create a list to store the processed data
    processed_data = []
    
    # Process each node in the route
    for node in route_nodes:
        processed_data.append({
            'fromICAO': data[0]['fromICAO'],
            'toICAO': data[0]['toICAO'],
            'type': node['type'],
            'ident': node['ident'],
            'lat': round(node['lat'], 8),
            'lon': round(node['lon'], 8),
            'alt': round(node['alt'], 8)
        })
    
    # Convert to DataFrame
    return pd.DataFrame(processed_data)

def process_json_file_for_metadata(filepath: str):
    # Load the JSON file
    with open(filepath, 'r') as f:
        data = json.load(f)

    processed_data = []
    if data is None or len(data) == 0:
        return pd.DataFrame()
    
    processed_data.append({
        'fromICAO': data[0]['fromICAO'],
        'toICAO': data[0]['toICAO'],
        'fromName': data[0]['fromName'],
        'toName': data[0]['toName'],
        'distance': data[0]['distance'],
        'maxAltitude': data[0]['maxAltitude'],
        'waypoints': data[0]['waypoints']
    })
    return pd.DataFrame(processed_data)

from pathlib import Path
from tqdm import tqdm
import os

if __name__ == "__main__":
    # List all JSON files in the routes directory
    routes_dir = Path("flight_schedules/routes")
    json_files = [f for f in routes_dir.glob("*.json")]
    print(f"Processing {len(json_files)} files")
    df_master = pd.DataFrame()
    df_metadata_master = pd.DataFrame()
    for file in tqdm(json_files):
        try:
            df = process_json_file_for_route(file)
            df_master = pd.concat([df_master, df], ignore_index=True)
            df_metadata = process_json_file_for_metadata(file)
            df_metadata_master = pd.concat([df_metadata_master, df_metadata], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            # Delete the error file
            os.remove(file)
            continue
    df_master.to_csv("flight_schedules/processed_data/route_db.csv", index=False)
    df_metadata_master.to_csv("flight_schedules/processed_data/route_metadata_db.csv", index=False)
