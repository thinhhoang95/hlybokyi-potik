import matplotlib.pyplot as plt
# Add the parent directory of this python file to search path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from path_prefix import PATH_PREFIX

# List to store the points clicked
points = []
# Initialize a variable to store the drawn flow
line = None

# Event handler for mouse clicks
def onclick(event):
    # Append the x and y coordinates of the click to the points list
    if event.xdata is not None and event.ydata is not None:  # Ensure it's a valid click inside the axes
        points.append((event.xdata, event.ydata))
        # Update the line on the plot with the new points
        update_plot()

# Function to update the plot with the new points and lines
def update_plot():
    global line  # We need to modify the line object in this function
    
    # If the line already exists, update its data
    if line:
        line.set_data([p[0] for p in points], [p[1] for p in points])
    else:
        # Create a new line if it doesn't exist yet
        line, = ax.plot([p[0] for p in points], [p[1] for p in points], marker='o', linestyle='-', color='b')
    
    plt.draw()  # Redraw the plot to show the updated line
    
# List all pickle files in the data/segments folder
pickle_files = [f for f in os.listdir(PATH_PREFIX + '/data/segments') if f.endswith('.pickle')]
print('Found', len(pickle_files), 'pickle files')

# Ask user to select a file
print('Select a file:')
for i, file in enumerate(pickle_files):
    print(f'{i}: {file}')
    
# Ask user to input the index of the file to load
file_index = int(input('Enter the index of the file to load: '))

file_path = PATH_PREFIX + '/data/segments/' + pickle_files[file_index]

# Read the segments from the pickle file
with open(file_path, 'rb') as f:
    segments_file = pickle.load(f)
    print('Segments file loaded')
    
seg_from_lat = segments_file['seg_from_lat']
seg_from_lon = segments_file['seg_from_lon']
seg_to_lat = segments_file['seg_to_lat']
seg_to_lon = segments_file['seg_to_lon']

# === Plot the segments ===
# Plot all the segments on a map

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

command = ''
flows = []

while command != 'd':
    print('Welcome. Flow specification tool developed by Thinh Hoang (dthoang@intuelle.com)')
    print('FLOW SPEC: Type "n" to specify a new flow, "d" when done')
    command = input()
    if command == 'd':
        break
    
    # Re-initialize the points and line to store user's selection
    line = None
    points = []

    # Create a new figure and axis with a map projection
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.set_aspect('auto')  # This allows the aspect ratio to adjust naturally

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Set the extent of the map to cover Europe
    ax.set_extent([-10, 30, 35, 60], crs=ccrs.PlateCarree())

    # Plot the segments
    for from_lat, from_lon, to_lat, to_lon in zip(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon):
        ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                color='red', linewidth=0.5, alpha=0.5, 
                transform=ccrs.Geodetic())
        
    # Plot all the flows in the list
    for flow in flows:
        ax.plot([p[0] for p in flow], [p[1] for p in flow], color='green', linewidth=1.5, transform=ccrs.Geodetic())

    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Setup the click event handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the plot
    plt.show()

    # After closing the plot, you can access the points
    print("Selected points:", points)
    
    # Add the flow to the list
    # Convert the points from np.float64 to float
    points = [(float(p[0]), float(p[1])) for p in points]
    flows.append(points)


print('Flow specification completed. Writing to file...')

# Create the flows directory if it doesn't exist
os.makedirs(PATH_PREFIX + '/data/flows/', exist_ok=True)

with open(PATH_PREFIX + '/data/flows/' + file_path.split('/')[-1].replace('.pickle', '.txt'), 'w') as f:
    for flow in flows:
        f.write(','.join(map(str, flow)) + '\n')

print('Flows written to file flows.txt')
print('Thank you for using the flow specification tool!')