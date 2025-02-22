import pandas as pd
from tqdm import tqdm

def lookup_tail_number(flight_number: list[str], timestamp: str) -> list[str]:
    # Load the ADS-B data from data/csv/timestamp.csv
    df_adsb = pd.read_csv(f"data/csv/{timestamp}.csv", header=None)
    # Add column names: time, icao24, lat, lon, hdg, callsign, alt
    df_adsb.columns = ['time', 'icao24', 'lat', 'lon', 'hdg', 'callsign', 'alt']

    # Convert callsign to lowercase and strip all whitespace
    df_adsb['callsign'] = df_adsb['callsign'].str.lower().str.strip()
    # Convert flight_number to lowercase and strip all whitespace
    flight_number = [str(flight_number).lower().strip() for flight_number in flight_number]

    # Find the matching flight using the flight number
    df_adsb = df_adsb[df_adsb['callsign'].isin(flight_number)]
    # Create a dictionary to store flight number -> icao24 mapping
    tail_numbers = {}
    
    # Get unique icao24 values for each flight number
    unique_icao24 = df_adsb.groupby('callsign')['icao24'].unique()
    
    # For each flight number, get the first icao24 value
    for flight_num in flight_number:
        if flight_num in unique_icao24.index:
            tail_numbers[flight_num] = unique_icao24[flight_num][0]
        else:
            pass
            
    return tail_numbers

if __name__ == "__main__":
    df_combined = pd.read_csv("flight_schedules/processed_data/combined/all_flights.csv")# Column names: time,airport,iata_number,airline_iata,codeshare_iata,departure_airport,scheduled_departure,actual_departure,arrival_airport,scheduled_arrival,actual_arrival,type
    df_airlines = pd.read_csv("airlines.csv") # Column names: Name,IATA,ICAO,Callsign,Country,Active
    # Remove rows with missing IATA or ICAO fields
    df_airlines = df_airlines.dropna(subset=['IATA', 'ICAO'])
    df_airlines.to_csv("airlines_cleaned.csv", index=False)
    # Add a column called 'icao24', default to None
    df_combined['icao24'] = None
    df_combined['airline_icao'] = None
    
    # Create airline IATA to ICAO mapping (converting IATA to lowercase for matching)
    airline_mapping = dict(zip(df_airlines['IATA'].str.lower(), df_airlines['ICAO']))
    
    # Map airline_iata to ICAO codes (airline_iata is already lowercase)
    df_combined['airline_icao'] = df_combined['airline_iata'].map(airline_mapping)
    
    # Create callsign by combining airline_icao with flight number
    # Extract flight number by removing first 2 characters (IATA code) from iata_number
    df_combined['callsign'] = df_combined['airline_icao'] + df_combined['iata_number'].str[2:]

    flight_numbers = df_combined['callsign'].unique()

    # Convert the time column to datetime
    df_combined['time'] = pd.to_datetime(df_combined['time'])

    # Converting the minimum time in df_combined to a timestamp
    min_time = df_combined['time'].min()
    # Make min_time the beginning of the hour
    min_time = min_time.replace(minute=0, second=0, microsecond=0)
    print(f'Min time: {min_time}')
    min_time_timestamp = int(min_time.timestamp())
    max_time = df_combined['time'].max()
    # Make max_time the end of the hour
    max_time = max_time.replace(minute=0, second=0, microsecond=0)
    print(f'Max time: {max_time}')
    max_time_timestamp = int(max_time.timestamp())

    # timestamps is a list of timestamps from min_time_timestamp to max_time_timestamp
    timestamps = list(range(min_time_timestamp, max_time_timestamp + 6*3600, 3600))
    print(f'There are {len(timestamps)} timestamps')

    # To match the tail number to each flight, the following steps will be taken:
    # We open ADS-B data from data/csv/timestamp.csv
    # We find the matching flight using the flight number
    # Then we extract the tail number, which is the icao24 field
    # We fill the icao24 field in the combined dataframe

    tail_numbers = {}

    for timestamp in tqdm(timestamps):
        tail_number_dict = lookup_tail_number(flight_numbers, timestamp)
        # Merge the tail_number_dict with the tail_numbers dictionary
        tail_numbers = {**tail_numbers, **tail_number_dict}

    # Example of tail_numbers:
    # {'ezy1234': 'a12345', 'ezy1235': 'a12346', 'ezy1236': 'a12347'}

    # Add the tail_numbers to the df_combined dataframe
    df_combined['icao24'] = df_combined['callsign'].str.lower().map(tail_numbers)
    # Save the df_combined dataframe to a new csv file
    df_combined.to_csv("flight_schedules/processed_data/combined/all_flights_with_tail_numbers.csv", index=False)


