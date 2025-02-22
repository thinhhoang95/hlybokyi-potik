import pandas as pd
from datetime import datetime

def parse_flight_json(json_data):
    # Handle both single record and array of records
    records = json_data if isinstance(json_data, list) else [json_data]
    flight_records = []

    for record in records:
        # Extract main fields
        flight_data = {
            'type': record.get('type'),
            'status': record.get('status'),
            'airline_name': record['airline']['name'],
            'airline_iata': record['airline']['iataCode'],
            'flight_number': record['flight']['number'],
            'iata_number': record['flight']['iataNumber'],
            'codeshare_airline': record['codeshared']['airline']['name'] if 'codeshared' in record else None,
            'codeshare_iata': record['codeshared']['airline']['iataCode'] if 'codeshared' in record else None
        }

        # Extract departure information
        departure = record['departure']
        flight_data.update({
            'departure_airport': departure['iataCode'],
            'departure_terminal': departure.get('terminal'),
            'departure_gate': departure.get('gate'),
            'departure_delay': departure.get('delay'),
            'scheduled_departure': datetime.fromisoformat(departure['scheduledTime'].lower()),
            'actual_departure': datetime.fromisoformat(departure['actualTime'].lower()) if 'actualTime' in departure else None
        })

        # Extract arrival information
        arrival = record['arrival']
        flight_data.update({
            'arrival_airport': arrival['iataCode'],
            'arrival_terminal': arrival.get('terminal'),
            'arrival_gate': arrival.get('gate'),
            'scheduled_arrival': datetime.fromisoformat(arrival['scheduledTime'].lower()),
            'actual_arrival': datetime.fromisoformat(arrival['actualTime'].lower()) if 'actualTime' in arrival else None
        })

        flight_records.append(flight_data)

    # Create DataFrame from all records
    df = pd.DataFrame(flight_records)
    
    # Convert datetime columns to string for better CSV compatibility
    datetime_cols = ['scheduled_departure', 'actual_departure', 'scheduled_arrival', 'actual_arrival']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None)
    
    return df

from glob import glob
import json

if __name__ == "__main__":
    print(f'Processing arrivals...')
    # List all json files in the raw_downloads/arrivals directory
    json_files = glob("flight_schedules/raw_downloads/arrivals/*.json")
    master_df = pd.DataFrame()
    for file in json_files:
        with open(file, "r") as f:
            json_data = json.load(f)
        df = parse_flight_json(json_data)
        master_df = pd.concat([master_df, df])
    
    # Save to csv
    master_df.to_csv(f"flight_schedules/processed_data/arrivals/all_arrivals.csv", index=False)

    print(f'Processing departures...')
    # List all json files in the raw_downloads/departures directory
    json_files = glob("flight_schedules/raw_downloads/departures/*.json")
    master_df = pd.DataFrame()
    for file in json_files:
        with open(file, "r") as f:
            json_data = json.load(f)
        df = parse_flight_json(json_data)
        master_df = pd.concat([master_df, df])
    # Save to csv
    master_df.to_csv(f"flight_schedules/processed_data/departures/all_departures.csv", index=False)
