from yaml import safe_load
from datetime import datetime, timedelta
from tqdm import tqdm
import requests
import json

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return safe_load(f)
    
def download_one_schedule(airport: str, date: str, mode: str):
    # Make a JSON request to the API at https://aviation-edge.com/v2/public/flightsHistory?code=SGN&type=arrival&date_from=2024-05-24&key=e5a949-95e253
    url = f"https://aviation-edge.com/v2/public/flightsHistory?code={airport}&type={mode}&date_from={date}&key=e5a949-95e253"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download schedule for {airport} on {date} with mode {mode}")
    return response.json()


def download_schedule(config: dict):
    iata_airports = config["iata_airports"]
    # iata_airports is a comma separated list of IATA airport codes
    # we need to convert it to a list of airport objects
    airports = [iata_code for iata_code in iata_airports.split(",")]

    date_from = config["date_from"]
    date_to = config["date_to"]

    # create a list of dates between date_from and date_to
    dates = [date_from + timedelta(days=i) for i in range(0, (date_to - date_from).days + 1)]

    # convert dates to strings
    dates = [date.strftime("%Y-%m-%d") for date in dates]

    print(f'Will download schedules for {len(dates)} dates and {len(airports)} airports')

    error_dates = []
    error_airports = []

    print(f'Downloading arrival schedules...')

    for date in tqdm(dates):
        for airport in airports:
            try:
                json_data = download_one_schedule(airport, date, "arrival")
                # Write to raw_downloads directory
                with open(f"flight_schedules/raw_downloads/arrivals/{airport}_{date}.json", "w") as f:
                    json.dump(json_data, f)
            except Exception as e:
                error_dates.append(date)
                error_airports.append(airport)
                print(f"Error downloading schedule for {airport} on {date}: {e}")

    print(f'Downloading departure schedules...')

    for date in tqdm(dates):
        for airport in airports:
            try:
                json_data = download_one_schedule(airport, date, "departure")
                # Write to raw_downloads directory
                with open(f"flight_schedules/raw_downloads/departures/{airport}_{date}.json", "w") as f:
                    json.dump(json_data, f)
            except Exception as e:
                error_dates.append(date)
                error_airports.append(airport)
                print(f"Error downloading schedule for {airport} on {date}: {e}")


if __name__ == "__main__":
    config = load_config("flight_schedules/config.yml")
    download_schedule(config)

    # json_data = download_one_schedule("PRG", "2024-05-25", "arrival")
    # with open(f"flight_schedules/raw_downloads/arrivals/PRG_2024-05-25.json", "w") as f:
    #     json.dump(json_data, f)


