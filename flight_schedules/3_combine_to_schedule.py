import pandas as pd

def combine_arrivals_and_departures(arrivals_df, departures_df):
    # For arrivals_df, add another column called 'type' with value 'arrival'
    arrivals_df['type'] = 'arrival'
    # For departures_df, add another column called 'type' with value 'departure'
    departures_df['type'] = 'departure'
    # Concatenate the two dataframes
    combined_df = pd.concat([arrivals_df, departures_df])
    # Only keep the columns: iata_number, airline_iata, codeshare_iata, departure_airport, scheduled_departure, actual_departure, arrival_airport, scheduled_arrival, actual_arrival, type
    combined_df = combined_df[['iata_number', 'airline_iata', 'codeshare_iata', 'departure_airport', 'scheduled_departure', 'actual_departure', 'arrival_airport', 'scheduled_arrival', 'actual_arrival', 'type']]
    # If codeshare_iata is empty or null, set it to the same value as airline_iata
    combined_df['codeshare_iata'] = combined_df['codeshare_iata'].fillna(combined_df['airline_iata'])
    # Add another column called 'time', if type is arrival, use scheduled_arrival, if type is departure, use scheduled_departure
    combined_df['time'] = combined_df.apply(lambda row: row['scheduled_arrival'] if row['type'] == 'arrival' else row['scheduled_departure'], axis=1)
    # Add another column called 'airport', if type is arrival, use arrival_airport, if type is departure, use departure_airport
    combined_df['airport'] = combined_df.apply(lambda row: row['arrival_airport'] if row['type'] == 'arrival' else row['departure_airport'], axis=1)
    # Sort by time, then by airport, then by iata_number
    combined_df = combined_df.sort_values(by=['time', 'airport', 'iata_number'])
    # Put the time, airport, and iata_number columns at the beginning of the dataframe
    combined_df = combined_df[['time', 'airport', 'iata_number', 'airline_iata', 'codeshare_iata', 'departure_airport', 'scheduled_departure', 'actual_departure', 'arrival_airport', 'scheduled_arrival', 'actual_arrival', 'type']]
    return combined_df

if __name__ == "__main__":
    arrivals_df = pd.read_csv("flight_schedules/processed_data/arrivals/all_arrivals.csv")
    departures_df = pd.read_csv("flight_schedules/processed_data/departures/all_departures.csv")
    combined_df = combine_arrivals_and_departures(arrivals_df, departures_df)
    # Save combine
    combined_df.to_csv("flight_schedules/processed_data/combined/all_flights.csv", index=False)
