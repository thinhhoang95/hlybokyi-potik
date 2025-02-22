import pandas as pd

df_airports = pd.read_csv("data/airports.csv")

def get_distance_between_airports(coord_from: tuple, coord_to: tuple) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        coord_from (tuple): Latitude and longitude of the first point (lat1, lon1)
        coord_to (tuple): Latitude and longitude of the second point (lat2, lon2)
    
    Returns:
        float: Distance between the two points in kilometers
    """
    import math

    # Earth's radius in kilometers
    R = 6371.0

    # Convert latitude and longitude to radians
    lat1, lon1 = math.radians(coord_from[0]), math.radians(coord_from[1])
    lat2, lon2 = math.radians(coord_to[0]), math.radians(coord_to[1])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Calculate distance
    distance = R * c

    return distance


def get_long_haul_assignment(df_airports: pd.DataFrame, iata_from: str, iata_to: str) -> tuple:
    # df_airports columns: country_code,region_name,iata,icao,airport,latitude,longitude

    try:
        # Convert iata_from and iata_to to uppercase
        iata_from = iata_from.upper()
        iata_to = iata_to.upper()

        # Lookup coordinates of the two airports
        coord_from = df_airports[df_airports["iata"] == iata_from]["latitude"].values[0], df_airports[df_airports["iata"] == iata_from]["longitude"].values[0]
        coord_to = df_airports[df_airports["iata"] == iata_to]["latitude"].values[0], df_airports[df_airports["iata"] == iata_to]["longitude"].values[0]

        # Convert coordinates to tuples
        coord_from = (coord_from[0], coord_from[1])
        coord_to = (coord_to[0], coord_to[1])

        # Get the distance between the two airports
        distance = get_distance_between_airports(coord_from, coord_to)

        # If the distance is smaller than 1500 km, it is a short haul flight, between 1500 and 4000 km, it is a medium haul flight, and if it is greater than 4000 km, it is a long haul flight
        if distance < 1500:
            return "short_haul", distance
        elif distance < 4000:
            return "medium_haul", distance
        else:
            return "long_haul", distance
    except Exception:
        print(f"Error assigning haul type to {iata_from} to {iata_to}")
        return "short_haul", -1
    

def process_combined_df(df_combined: pd.DataFrame, save_path: str) -> pd.DataFrame:
    # df_combined columns: time,airport,iata_number,airline_iata,codeshare_iata,departure_airport,scheduled_departure,actual_departure,arrival_airport,scheduled_arrival,actual_arrival,type

    # Get the long haul assignment for each flight
    from tqdm import tqdm
    tqdm.pandas()
    df_combined["haul_type"], df_combined["distance"] = zip(*df_combined.progress_apply(lambda row: get_long_haul_assignment(df_airports, row["departure_airport"], row["arrival_airport"]), axis=1))

    # Save the combined dataframe to a csv file
    df_combined.to_csv(save_path, index=False)

    return df_combined

if __name__ == "__main__":
    print('Assigning haul type to all flights... This might take a while...')
    df_combined = pd.read_csv("flight_schedules/processed_data/combined/all_flights.csv")
    df_combined = process_combined_df(df_combined, "flight_schedules/processed_data/combined/all_flights_with_haul_type.csv")

