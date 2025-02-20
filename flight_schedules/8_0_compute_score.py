import numpy as np

def get_haul_type_weight(arrival_haul_type: str, departure_haul_type: str) -> float:
    W = np.array([[0.8, 0.2],
                  [0.4, 0.6]])
    if arrival_haul_type == "short_haul" and departure_haul_type == "short_haul":
        w = W[0, 0]
    elif arrival_haul_type == "short_haul" and departure_haul_type == "long_haul":
        w = W[0, 1]
    elif arrival_haul_type == "long_haul" and departure_haul_type == "short_haul":
        w = W[1, 0]
    elif arrival_haul_type == "long_haul" and departure_haul_type == "long_haul":
        w = W[1, 1]
    return w

gamma_params = {
    'short_haul_to_short_haul': {
        'a': 1.9599999999999997,
        'scale': 0.17857142857142858,
        'loc': 0.65
    },
    'short_haul_to_long_haul': {
        'a': 2.5599999999999987,
        'scale': 0.15625000000000003,
        'loc': 1.85
    },

    'long_haul_to_short_haul': {
        'a': 4.456790123456792,
        'scale': 0.10657894736842101,
        'loc': 1.25
    },

    'long_haul_to_long_haul': {
        'a': 4.0,
        'scale': 0.25,
        'loc': 2.25
    }

}

from scipy.stats import gamma

def get_score(t_conn: float, arrival_haul_type: str, departure_haul_type: str) -> float:
    """
    Calculate the connection score based on connection time and haul types.

    This function computes a score for a flight connection using a weighted gamma distribution
    probability density function. The score reflects the likelihood of a connection based on
    the time between flights and their respective haul types.

    Args:
        t_conn (float): Connection time between flights.
        arrival_haul_type (str): Haul type of the arriving flight (e.g., 'short_haul', 'long_haul').
        departure_haul_type (str): Haul type of the departing flight.

    Returns:
        float: A log-transformed score representing the connection's probability,
               weighted by haul type and adjusted for connection time. The larger score, the more likely the connection.

    Notes:
        - Medium haul flights are converted to short haul for calculation purposes.
        - The score uses pre-defined gamma distribution parameters for different haul type combinations.
    """
    # convert medium_haul to short_haul since they are typically exploited by narrow body aircraft
    if arrival_haul_type == "medium_haul":
        arrival_haul_type = "short_haul"
    if departure_haul_type == "medium_haul":
        departure_haul_type = "short_haul"

    w = get_haul_type_weight(arrival_haul_type, departure_haul_type)


    params = gamma_params[f"{arrival_haul_type}_to_{departure_haul_type}"]
    score = np.log(w * gamma.pdf(t_conn, **params))
    return score

import pandas as pd

def get_score_matrix_for_airline(df_combined: pd.DataFrame) -> np.ndarray:
    # df_combined columns: time,airport,iata_number,airline_iata,codeshare_iata,departure_airport,scheduled_departure,actual_departure,arrival_airport,scheduled_arrival,actual_arrival,type,haul_type,distance
    airports = df_combined["departure_airport"].unique()
    # concatenate airports with df_combined["arrival_airport"]
    airports = np.concatenate([airports, df_combined["arrival_airport"].unique()])
    airports = np.unique(airports)

    print(f'Found {len(airports)} airports')

    # airports = ['cdg'] # for testing

    # For each airport
    for airport in airports:
        print(f"Processing airport: {airport}")
        airlines = df_combined[df_combined["airport"] == airport]["codeshare_iata"].unique()
        # airlines = ['af'] # for testing
        print(f'Found {len(airlines)} airlines')
        for airline_index, airline in enumerate(airlines):
            # if not (airline == 'qr' and airport == 'lhr'): # for debugging why qr lhr does not show up in score matrix???
            #     continue
            print(f"Processing airline  {airline} at {airport} ({airline_index+1}/{len(airlines)})")

            df_airline = df_combined[df_combined["codeshare_iata"] == airline]
            df_airport = df_airline[df_airline["airport"] == airport]

            df_arrivals = df_airport[df_airport["type"] == "arrival"]
            df_departures = df_airport[df_airport["type"] == "departure"]

            arrival_times = df_arrivals["scheduled_arrival"]
            arrival_haul_types = df_arrivals["haul_type"].values
            departure_times = df_departures["scheduled_departure"]
            departure_haul_types = df_departures["haul_type"].values

            # Convert times to datetime
            arrival_times = pd.to_datetime(arrival_times)
            departure_times = pd.to_datetime(departure_times)

            # Initialize score matrix
            n_arrivals = len(arrival_times)
            n_departures = len(departure_times)
            if n_arrivals == 0 or n_departures == 0:
                print(f"No arrivals or departures for {airport} {airline}")
                continue

            S = np.zeros((n_arrivals, n_departures))

            # Compute connection times and scores efficiently using broadcasting
            arrival_times_matrix = arrival_times.values.reshape(-1, 1)  # Shape: (n_arrivals, 1)
            departure_times_matrix = departure_times.values.reshape(1, -1)  # Shape: (1, n_departures)
            
            # Connection times in hours
            connection_times = (departure_times_matrix - arrival_times_matrix).astype('timedelta64[h]').astype(float)

            # Calculate scores for each valid connection
            from tqdm import tqdm
            for i in tqdm(range(n_arrivals), desc="Processing arrivals"):
                for j in range(n_departures):
                    if connection_times[i,j] > 0 and connection_times[i,j] < 8:  # Only consider positive connection times
                        S[i,j] = get_score(
                            connection_times[i,j],
                            arrival_haul_types[i],
                            departure_haul_types[j]
                        )
                    else:
                        S[i,j] = float('-inf')  # Invalid connection (e.g. connection time is negative or greater than 8 hours)

            # Save S to a file
            np.save(f"flight_schedules/processed_data/score_matrices/{airport}_{airline}.npy", S)


import os

if __name__ == "__main__":
    df_combined = pd.read_csv("flight_schedules/processed_data/combined/all_flights_with_haul_type_collapsed.csv")
    # Create a folder for score matrices
    os.makedirs("flight_schedules/processed_data/score_matrices", exist_ok=True)
    S = get_score_matrix_for_airline(df_combined)