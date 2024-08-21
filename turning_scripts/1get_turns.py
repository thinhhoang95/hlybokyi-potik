from get_turns import get_turning_points, plot_changepoints, TurnAndRise
from potters import plot_df
import pandas as pd
import uuid
from tqdm import tqdm

import os, sys
# add parent directory to sys.path to import path_prefix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_prefix import PATH_PREFIX

def create_dirs() -> None: 
    os.makedirs(PATH_PREFIX + "/data/wp", exist_ok=True)

def generate_waypoint_names() -> str: 
    # Generate a random uuid of 10 characters
    return str(uuid.uuid4())[:10]

def process_sample_file(df_flight: pd.DataFrame):
    turns_df:pd.DataFrame = pd.DataFrame(columns=['id', 'waypoint_name', 'waypoint_lat', 'waypoint_lon', 'waypoint_alt', 'waypoint_time', 'waypoint_speed'])
    rises_df:pd.DataFrame = pd.DataFrame(columns=['id', 'waypoint_name', 'waypoint_lat', 'waypoint_lon', 'waypoint_alt', 'waypoint_time', 'waypoint_speed'])
    # Get the unique ids
    id_list = df_flight['id'].unique()
    for id in tqdm(id_list):
        df_flight_id = df_flight[df_flight['id'] == id]
        tr:TurnAndRise = get_turning_points(df_flight_id)
        # Create a temporary DataFrame with the information from tr object
        
        temp_df_turns = pd.DataFrame({
            'id': [id] * len(tr['tp_time']),
            'waypoint_name': [generate_waypoint_names() for _ in range(len(tr['tp_time']))],
            'waypoint_lat': tr['tp_lat'],
            'waypoint_lon': tr['tp_lon'],
            'waypoint_alt': tr['tp_alt'],
            'waypoint_time': tr['tp_time'],
            'waypoint_speed': tr['tp_vel']
        })

        temp_df_rises = pd.DataFrame({
            'id': [id] * len(tr['tp_time']),
            'waypoint_name': [generate_waypoint_names() for _ in range(len(tr['tp_time']))],
            'waypoint_lat': tr['tp_lat'],
            'waypoint_lon': tr['tp_lon'],
            'waypoint_alt': tr['tp_alt'],
            'waypoint_time': tr['tp_time'],
            'waypoint_speed': tr['tp_vel']
        })


        
        # Concatenate the temporary DataFrame with turns_df and rises_df
        turns_df = pd.concat([turns_df, temp_df_turns], ignore_index=True)
        rises_df = pd.concat([rises_df, temp_df_rises], ignore_index=True)

    return turns_df, rises_df

if __name__ == "__main__":
    create_dirs()
    df_flight = pd.read_csv(PATH_PREFIX + "/data/sample/sample.csv")
    turns_df, rises_df = process_sample_file(df_flight)
    # Save tr_df to a csv file
    turns_df.to_csv(PATH_PREFIX + "/data/wp/sample.turns.csv", index=False)
    rises_df.to_csv(PATH_PREFIX + "/data/wp/sample.rises.csv", index=False)