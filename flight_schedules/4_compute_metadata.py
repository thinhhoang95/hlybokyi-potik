import pandas as pd
import matplotlib.pyplot as plt
def derive_routes_metadata(df_all, df_airports):
    # convert df_airports iata to lowercase
    df_airports['iata'] = df_airports['iata'].str.lower().str.strip()
    # Add a column called 'OD', concatenating the departure_airport and arrival_airport columns
    df_all['departure_airport_icao'] = df_all['departure_airport'].map(df_airports.set_index('iata')['icao'])
    df_all['arrival_airport_icao'] = df_all['arrival_airport'].map(df_airports.set_index('iata')['icao'])
    df_all['OD'] = df_all['departure_airport_icao'] + '-' + df_all['arrival_airport_icao']
    # Return unique OD values
    return df_all['OD'].unique()

if __name__ == "__main__":
    df_all = pd.read_csv("flight_schedules/processed_data/combined/all_flights.csv")
    df_airports = pd.read_csv("data/airports.csv") # columns: country_code,region_name,iata,icao,airport,latitude,longitude
    od_pairs = derive_routes_metadata(df_all, df_airports)
    print(f'There are {len(df_all)} flights')
    print(f'There are {len(od_pairs)} unique OD pairs')
    # Write all OD pairs to a CSV file
    # Write OD pairs to a CSV file
    pd.DataFrame(od_pairs, columns=['OD']).to_csv("flight_schedules/processed_data/od_pairs.csv", index=False)
    # Find the distribution of arrival times
    df_all['arrival_time'] = pd.to_datetime(df_all['scheduled_arrival'])
    df_all['arrival_time'] = df_all['arrival_time'].dt.hour
    # Plot the distribution of arrival times
    plt.hist(df_all['arrival_time'], bins=48, edgecolor='black')
    plt.xlabel('Arrival Time (Hour)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Arrival Times')
    plt.show()

    # Plot the distribution of departure times
    df_all['departure_time'] = pd.to_datetime(df_all['scheduled_departure'])
    df_all['departure_time'] = df_all['departure_time'].dt.hour
    plt.hist(df_all['departure_time'], bins=48, edgecolor='black')
    plt.xlabel('Departure Time (Hour)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Departure Times')
    plt.show()
