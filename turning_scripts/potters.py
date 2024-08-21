import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_df(df: pd.DataFrame, ident = None) -> None:
    """Plot the ADS-B DataFrame

    Args:
        df (pd.DataFrame): Dataframe with latitude and longitude columns
    """
    # Check if the dataframe has the required columns
    if 'ident' not in df.columns:
        df['ident'] = (df['callsign'].str.strip()+'_'+df['icao24'].str.strip())

    # If ident is specified, filter the dataframe for the specific ident
    if ident is not None:
        df = df[df['ident'] == ident]

    plt.figure(figsize=(6,6))
    plt.subplot(2, 2, 1)
    plt.plot(df['lon'], df['lat'])
    plt.plot(df['lon'].iloc[0], df['lat'].iloc[0], 'rx')
    plt.plot(df['lon'].iloc[-1], df['lat'].iloc[-1], 'go')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flight Path')
    # Set aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(2, 2, 2)
    plt.plot(df['lastposupdate'], df['geoaltitude'])
    plt.xlabel('Time')
    plt.ylabel('Altitude')
    plt.title('Altitude vs Time')

    plt.subplot(2, 2, 3)
    plt.plot(df['lastposupdate'], df['heading'])
    plt.xlabel('Time')
    plt.ylabel('Heading')
    plt.title('Heading vs Time')

    plt.subplot(2, 2, 4)
    plt.plot(df['lastposupdate'], df['velocity'])
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Time')

    plt.tight_layout()

    plt.show()
