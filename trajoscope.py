from matplotlib import pyplot as plt
import pandas as pd

import os
import sys

# Get the current directory
current_dir = os.path.dirname(os.path.abspath('__file__'))
print(current_dir)

# Get the parent directory
parent_dir = os.path.dirname(current_dir)
print(parent_dir)

# Add the parent directory to sys.path
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'turning_scripts'))
sys.path.append(os.path.join(current_dir, 'cleaning_script'))

from data_preambles import dtypes_no_id, col_names, csv_to_exclude, catalog_col_names
from path_prefix import PATH_PREFIX

def plot_trajectory(timestamp, callsigns = [], ax=None,
                    show_turn_overlay = False):
    if ax is None:
        ax = plt.gca()
    
    df = pd.read_csv(f'{PATH_PREFIX}/data/csv/{timestamp}.csv')
    df.columns = col_names

    df['callsign'] = df['callsign'].str.strip()
    # Get the global time range for consistent coloring
    time_min = df['time'].min()
    time_max = df['time'].max()
    
    # Create a list to store all scatter plots
    scatter_plots = []
    for callsign in callsigns:
        df_callsign = df[df['callsign'] == callsign]
        scatter = ax.scatter(df_callsign['lon'], df_callsign['lat'], 
                           c=df_callsign['time'], cmap='viridis', 
                           label=callsign, s=5, alpha=0.5,
                           vmin=time_min, vmax=time_max)
        scatter_plots.append(scatter)

        if show_turn_overlay:
            from get_turns import get_turning_points
            from cleaning_script import clean_by_speed as cleaner
            # Add a unique identifier to the dataframe
            df_callsign['id'] = df_callsign['callsign'] + df_callsign['icao24']
            # Clean the dataframe
            df_callsign = cleaner.clean_trajectory(df_callsign)
            turns = get_turning_points(df_callsign)
            plt.plot(turns['tp_lon'], turns['tp_lat'], c='red', linestyle='-', linewidth=1)
    
    # Add a single colorbar using the last scatter plot
    if scatter_plots:
        plt.colorbar(scatter_plots[-1], ax=ax, label='Time')
    ax.legend()
    plt.show()