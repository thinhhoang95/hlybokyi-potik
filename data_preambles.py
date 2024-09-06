# Not used anymore
# dtypes = {
#     'time': 'float64',
#     'icao24': 'str',
#     'lat': 'float64',
#     'lon': 'float64',
#     'heading': 'float64',
#     'callsign': 'str',
#     'geoaltitude': 'float64',
#     'id': 'str'
# }

dtypes_no_id = {
    'time': 'float64',
    'icao24': 'str',
    'lat': 'float64',
    'lon': 'float64',
    'heading': 'float64',
    'callsign': 'str',
    'geoaltitude': 'float64'
}

col_names = ['time', 'icao24', 'lat', 'lon', 'heading', 'callsign', 'geoaltitude']

csv_to_exclude = ['dangling.buffer.csv', 'dangling.csv', 'catalog.csv', 'dangling_explicit.csv']

catalog_col_names = ['id', 'from_timestamp', 'to_timestamp']