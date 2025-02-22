import pandas as pd

def collapse_codeshare(df_combined: pd.DataFrame):
    # df_combined columns: time,airport,iata_number,airline_iata,codeshare_iata,departure_airport,scheduled_departure,actual_departure,arrival_airport,scheduled_arrival,actual_arrival,type,haul_type,distance

    # add a column called "codeshare_code" which is the concatenation of "codeshare_iata", "departure_airport", "arrival_airport", "scheduled_departure", "scheduled_arrival"
    df_combined["codeshare_code"] = df_combined["codeshare_iata"] + "_" + df_combined["departure_airport"] + "_" + df_combined["arrival_airport"] + "_" + df_combined["scheduled_departure"].astype(str) + "_" + df_combined["scheduled_arrival"].astype(str)
    # drop duplicates
    df_combined = df_combined.drop_duplicates(subset=["codeshare_code"])
    # drop the codeshare_code column
    df_combined = df_combined.drop(columns=["codeshare_code"])
    return df_combined

if __name__ == "__main__":
    df_combined = pd.read_csv("flight_schedules/processed_data/combined/all_flights_with_haul_type.csv")
    print(f'Original number of flights: {len(df_combined)}')
    df_combined = collapse_codeshare(df_combined)
    print(f'Collapsed number of flights: {len(df_combined)}')
    df_combined.to_csv("flight_schedules/processed_data/combined/all_flights_with_haul_type_collapsed.csv", index=False)




