import pandas as pd
import numpy as np
from tqdm import tqdm

import os 

def derive_conn_time(df_master: pd.DataFrame):
    # Initialize connection_time column with NaN
    df_master['connection_time'] = np.nan
    
    # df_combined columns: time,airport,iata_number,airline_iata,codeshare_iata,departure_airport,scheduled_departure,actual_departure,arrival_airport,scheduled_arrival,actual_arrival,type,haul_type,distance
    airports = df_master["departure_airport"].unique()
    # concatenate airports with df_master["arrival_airport"]

    airports = np.concatenate([airports, df_master["arrival_airport"].unique()])
    airports = np.unique(airports)

    print(f'Found {len(airports)} airports')

    # airports = ['cdg'] # for testing
    

    # For each airport
    for airport in airports:
        print(f"Processing airport: {airport}")
        airlines = df_master[df_master["airport"] == airport]["codeshare_iata"].unique()
        # airlines = ['af'] # for testing
        print(f'Found {len(airlines)} airlines')
        for airline_index, airline in tqdm(enumerate(airlines), desc="Processing airlines"):
            # print(f"Processing airline  {airline} at {airport} ({airline_index+1}/{len(airlines)})")
            df_airline = df_master[df_master["codeshare_iata"] == airline]
            df_airport = df_airline[df_airline["airport"] == airport]


            df_arrivals = df_airport[df_airport["type"] == "arrival"]
            df_departures = df_airport[df_airport["type"] == "departure"]

            # We load the corresponding assignment file
            assignments_file = f"flight_schedules/processed_data/assignments/{airport}_{airline}.npy.csv"
            # Check if the file exists
            if not os.path.exists(assignments_file):
                # print(f"Assignments file {assignments_file} does not exist")
                continue
            df_assignments = pd.read_csv(assignments_file)


            # Convert datetime strings to datetime objects
            arrivals_times = pd.to_datetime(df_arrivals['scheduled_arrival'])
            departures_times = pd.to_datetime(df_departures['scheduled_departure'])

            # Calculate connection times for matched flights
            for _, row in df_assignments.iterrows():
                arrival_idx = row['arrival']
                departure_idx = row['departure']
                
                # Get the arrival and departure times
                arrival_time = arrivals_times.iloc[arrival_idx]
                departure_time = departures_times.iloc[departure_idx]
                
                # Calculate time difference in hours
                time_diff = (departure_time - arrival_time).total_seconds() / 3600
                
                # Get the original index from df_arrivals to update df_master
                original_arrival_idx = df_arrivals.index[arrival_idx]
                df_master.loc[original_arrival_idx, 'connection_time'] = time_diff

    # Save the updated df_master to a new CSV file
    df_master.to_csv("flight_schedules/processed_data/combined/all_flights_with_haul_type_and_conn_time.csv", index=False)

if __name__ == "__main__":
    df_master = pd.read_csv("flight_schedules/processed_data/combined/all_flights_with_haul_type_collapsed.csv")
    # df_assignments = pd.read_csv("flight_schedules/processed_data/assignments/all_flights_with_haul_type_and_assignments.csv")
    derive_conn_time(df_master)

