from matplotlib import pyplot as plt

import sys
import os

import pickle
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import yaml


# Add the parent directory of this python file to the search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the segment_clustering_base folder to the search path
sys.path.append(os.path.join(parent_dir, 'segment_clustering_base'))
sys.path.append(parent_dir)

from path_prefix import PATH_PREFIX

from segment_clustering_base.sim_model import get_similarity_matrix, get_similarity_matrix_polyline
from segment_clustering_base.dsa import TDFParams


def load_segments():
    """Load the segments from the segments folder.
    Returns:
        seg_from_lat (1D np.ndarray): the from latitude of the segments
        seg_from_lon (1D np.ndarray): the from longitude of the segments
        seg_to_lat (1D np.ndarray): the to latitude of the segments
        seg_to_lon (1D np.ndarray): the to longitude of the segments
        filename (str): the filename of the segments (without the .segments.pickle extension)
    """
    # List all the .pickle files in 'data/segments' folder
    
    # Get the path to the segments folder
    segments_folder = os.path.join(PATH_PREFIX, 'data', 'segments')
    
    # List all .pickle files in the segments folder
    pickle_files = [f for f in os.listdir(segments_folder) if f.endswith('.segments.pickle')]
    
    print(f"Found {len(pickle_files)} pickle files of segments:")
    for i, file in enumerate(pickle_files):
        print(f"{i + 1}. {file}")

    # Ask the user to select a file
    file_idx = int(input("Enter the number of the file you want to use: ")) - 1
    file_path = os.path.join(segments_folder, pickle_files[file_idx])

    # Load the pickle file
    with open(file_path, 'rb') as f:
        segments = pickle.load(f)

    seg_from_lat = segments['seg_from_lat']
    seg_from_lon = segments['seg_from_lon']
    seg_to_lat = segments['seg_to_lat']
    seg_to_lon = segments['seg_to_lon']

    # Convert deque to np.array
    seg_from_lat = np.array(seg_from_lat)
    seg_from_lon = np.array(seg_from_lon)
    seg_to_lat = np.array(seg_to_lat)
    seg_to_lon = np.array(seg_to_lon)

    print(f"Loaded {len(seg_from_lat)} segments from {file_path}")

    # Get the filename from file_path
    filename = os.path.splitext(os.path.basename(file_path))[0]
    filename = filename.replace('.segments', '')
    return seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, filename

def flow_specification_interface(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, polylines, show_labels=False, filter=None):
    # Create a new figure and axis with a map projection
    fig, ax = plt.subplots(figsize=(4, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_aspect('auto')  # This allows the aspect ratio to adjust naturally

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Set the extent of the map to cover Europe
    ax.set_extent([-10, 40, 28, 75], crs=ccrs.PlateCarree())

    i = 0
    for from_lat, from_lon, to_lat, to_lon in zip(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon):
        if filter is not None and i not in filter:
            continue # Skip the segment because it is not in the filter
        ax.plot([from_lon, to_lon], [from_lat, to_lat],
                color='red', linewidth=1., alpha=0.2,
                transform=ccrs.Geodetic())

        # Add a text label at the midpoint of the segment
        if show_labels:
            mid_lat = (from_lat + to_lat) / 2
            mid_lon = (from_lon + to_lon) / 2
            # label is the index of the segment
            label = np.where((seg_from_lat == from_lat) & (seg_from_lon == from_lon) & 
                             (seg_to_lat == to_lat) & (seg_to_lon == to_lon))[0][0]
            ax.text(mid_lon, mid_lat, str(label), transform=ccrs.Geodetic(), 
                    fontsize=8, ha='center', va='center', color='black')

        # Add a marker x at the end of the segment
        ax.plot([to_lon], [to_lat], marker='x', color='red', transform=ccrs.Geodetic(), alpha=0.2)

    # Draw the polylines with green color on top
    for polyline in polylines:
        x, y = zip(*polyline)
        ax.plot(x, y, 'g-', linewidth=2, transform=ccrs.PlateCarree())

    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Initialize an empty list to store the polyline points
    polyline = []
    line, = ax.plot([], [], 'b-', linewidth=2, transform=ccrs.PlateCarree())

    # Create a text object for instructions
    instruction_text = ax.text(0.5, 0.02, "Click to add points. Press 'Enter' to finish.", 
                               ha='center', va='bottom', transform=ax.transAxes)

    def onclick(event):
        if event.inaxes != ax:
            return
        lon, lat = event.xdata, event.ydata
        polyline.append((lon, lat))
        x, y = zip(*polyline)
        line.set_data(x, y)
        
        # Update only the polyline
        ax.draw_artist(line)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

    def onkey(event):
        if event.key == 'enter':
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
            instruction_text.set_visible(False)
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show()

    return polyline
    
def write_polylines_to_file(polylines, filename):
    filename = filename + '.polylines.pickle'
    # Create the flows folder if it doesn't exist
    flows_folder = os.path.join(PATH_PREFIX, 'data', 'flows')
    os.makedirs(flows_folder, exist_ok=True)

    # Write the polylines to a file
    with open(os.path.join(flows_folder, filename), 'wb') as f:
        pickle.dump(polylines, f)

    print(f"Wrote {len(polylines)} polylines to {os.path.join(flows_folder, filename)}")


def load_polylines_from_file(filename): # filename without .polylines.pickle
    filename = filename + '.polylines.pickle'
    flows_folder = os.path.join(PATH_PREFIX, 'data', 'flows')
    with open(os.path.join(flows_folder, filename), 'rb') as f:
        polylines = pickle.load(f)
    return polylines

from scipy.spatial.distance import cdist

def resample_polyline(polyline, resolution_degrees):
    """Resample the polyline to have points spaced approximately resolution_degrees apart.

    Args:
        polyline (list): List of (lon, lat) tuples representing the polyline.
        resolution_degrees (float): Desired spacing between points in degrees.

    Returns:
        np.array: Resampled polyline as a 2D numpy array.
    """
    polyline = np.array(polyline)
    
    # Calculate the total length of the polyline
    total_length = np.sum(np.sqrt(np.sum(np.diff(polyline, axis=0)**2, axis=1)))
    
    # Calculate the number of points for the resampled polyline
    num_points = int(total_length / resolution_degrees) + 1

    print(f'Number of resampled points for the polyline: {num_points}')
    
    # Calculate the cumulative distance along the polyline
    distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Add starting point
    
    # Create evenly spaced points along the total distance
    even_distances = np.linspace(0, distances[-1], num_points)
    
    # Interpolate new points
    resampled_polyline = np.column_stack([
        np.interp(even_distances, distances, polyline[:, 0]),
        np.interp(even_distances, distances, polyline[:, 1])
    ])
    
    return resampled_polyline

def get_long_enough_segments(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, min_length, neighbor_indices):
    """
    Get the indices of the segments that are at least min_length long.

    Parameters:
        seg_from_lat (1D np.ndarray): the from latitude of the segments
        seg_from_lon (1D np.ndarray): the from longitude of the segments
        seg_to_lat (1D np.ndarray): the to latitude of the segments
        seg_to_lon (1D np.ndarray): the to longitude of the segments
        min_length (float): the minimum length of the segments in kilometers

    Returns:
        np.ndarray: the indices of the segments that are at least min_length long
    """
    indices = []

    # Compute the length of each segment
    segment_lengths = np.sqrt(np.square(seg_from_lat - seg_to_lat) + np.square(seg_from_lon - seg_to_lon))

    # Get the indices of the segments that are at least min_length long
    indices = np.where(segment_lengths >= min_length)[0]

    # Get the intersection of the indices with the neighbor_indices
    indices = np.intersect1d(indices, neighbor_indices)

    return indices

def get_epsilon_neighborhood(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, polyline, epsilon):
    """Get the indices of the segments within epsilon distance of the polyline."""
    indices = []

    # Convert polyline to numpy array
    polyline_points = np.array(polyline)

    # Create arrays for segment start and end points
    seg_start_points = np.column_stack((seg_from_lon, seg_from_lat))
    seg_end_points = np.column_stack((seg_to_lon, seg_to_lat))

    # Compute distances from polyline points to segment start and end points
    distances_start = cdist(polyline_points, seg_start_points)
    distances_end = cdist(polyline_points, seg_end_points)

    # Find segments where either start or end point is within epsilon distance
    within_epsilon = np.logical_or(
        np.min(distances_start, axis=0) <= epsilon,
        np.min(distances_end, axis=0) <= epsilon
    )

    # Get indices of segments within epsilon distance
    indices = np.where(within_epsilon)[0]

    return indices



def flow_refinement_interface(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, polyline, neighbor_indices):
    # Create a new figure and axis with a map projection
    fig, ax = plt.subplots(figsize=(4, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_aspect('auto')  # This allows the aspect ratio to adjust naturally

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Set the extent of the map to cover Europe
    ax.set_extent([-10, 40, 28, 75], crs=ccrs.PlateCarree())

    elected_seg_from_lat = seg_from_lat[neighbor_indices]
    elected_seg_from_lon = seg_from_lon[neighbor_indices]
    elected_seg_to_lat = seg_to_lat[neighbor_indices]
    elected_seg_to_lon = seg_to_lon[neighbor_indices]

    # List to store the plot objects
    segment_lines = []
    segment_markers = []

    # List to store relegated segments
    relegated_segments = []

    # List to store the history of actions for undo functionality
    action_history = []

    for i, (from_lat, from_lon, to_lat, to_lon) in enumerate(zip(elected_seg_from_lat, elected_seg_from_lon, elected_seg_to_lat, elected_seg_to_lon)):
        line = ax.plot([from_lon, to_lon], [from_lat, to_lat],
                       color='red', linewidth=1., alpha=0.2,
                       transform=ccrs.Geodetic())[0]
        marker = ax.plot([to_lon], [to_lat], marker='x', color='red', 
                         transform=ccrs.Geodetic(), alpha=0.2)[0]
        
        segment_lines.append(line)
        segment_markers.append(marker)

    # Draw the polyline with green color on top
    x, y = zip(*polyline)
    ax.plot(x, y, 'g-', linewidth=2, transform=ccrs.PlateCarree(), alpha=0.2)

    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    again_flag = False # to indicate if the user wants to sample more segments to label for this flow

    def on_click(event):
        if event.inaxes == ax:
            for i, (line, marker) in enumerate(zip(segment_lines, segment_markers)):
                if line.contains(event)[0] or marker.contains(event)[0]:
                    line.set_visible(False)
                    marker.set_visible(False)
                    relegated_segments.append(neighbor_indices[i])
                    action_history.append(('remove', i))
                    fig.canvas.draw()
                    break

    def on_key(event):
        nonlocal relegated_segments, again_flag  # Add this line to access relegated_segments
        if event.key == 'z' and action_history:
            action, index = action_history.pop()
            if action == 'remove':
                segment_lines[index].set_visible(True)
                segment_markers[index].set_visible(True)
                relegated_segments.remove(neighbor_indices[index])
                fig.canvas.draw()
        elif event.key == 's':
            plt.close(fig)
            relegated_segments = None  # Set relegated_segments to None
        elif event.key == 'a':
            again_flag = True # the user wants to sample more segments to label for this flow

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    return relegated_segments, again_flag

import random
import string

def generate_random_id(length=10):
    """
    Generate a random string of characters for use as a filename.
    
    Args:
        length (int): The length of the random string. Default is 10.
    
    Returns:
        str: A random string of lowercase letters and digits.
    """
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def create_training_data_folder():
    """
    Create a 'train' folder if it doesn't exist.
    """
    train_folder = os.path.join(PATH_PREFIX, 'data', 'c1_train')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        print(f"Created 'c1_train' folder at {train_folder}")
    else:
        print(f"'c1_train' folder already exists at {train_folder}")

    flows_folder = os.path.join(train_folder, 'flows')
    segments_folder = os.path.join(train_folder, 'segments')

    for folder in [flows_folder, segments_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")

import csv

# def write_admitted_segments_to_file(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, admitted_segments_indices, polyline, filename):
#     # Specify the output CSV file name
#     flow_id = generate_random_id()
#     output_file_flow = os.path.join(PATH_PREFIX, 'data', 'c1_train', 'flows', f'{filename}_{flow_id}.csv')

#     # Write the polyline data to the CSV file
#     with open(output_file_flow, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
        
#         # Write header
#         writer.writerow(['x', 'y'])
        
#         # Write each joint (row) of the polyline
#         for joint in polyline:
#             writer.writerow(joint)

#     # Write the admitted segments to a file
#     output_file_segments = os.path.join(PATH_PREFIX, 'data', 'c1_train', 'segments', f'{filename}_{flow_id}.csv')
#     with open(output_file_segments, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
        
#         # Write header
#         writer.writerow(['from_lat', 'from_lon', 'to_lat', 'to_lon'])
        
#         # Write each segment (row)
#         for i in admitted_segments_indices:
#             writer.writerow([seg_from_lat[i], seg_from_lon[i], seg_to_lat[i], seg_to_lon[i]])

#     print(f"Wrote admitted segments to {output_file_segments}")

def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(i) for i in obj]
    else:
        return obj

def write_label_data_to_yaml_file(filename: str, admitted_indices: list, labelled_indices: list, label: list):

    # Specify the output YAML file name
    output_file = os.path.join(PATH_PREFIX, 'data', 'c1_train', 'flows', f'{filename}.yaml')

    # Write the data to the YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump({
            'admitted_indices': numpy_to_list(admitted_indices),
            'labelled_indices': numpy_to_list(labelled_indices),
            'label': numpy_to_list(label)
        }, yaml_file)

    print(f"Wrote label data to {output_file}")

def subtract_lists(list1, list2):
    return [item for item in list1 if item not in set(list2)]

from segment_clustering_base.spectral_clustereuse import plot_segments_with_labels

if __name__ == '__main__':
    print('Data Generation Interface for Flow Identification using Bootstrapped Graph Neural Networks')
    print('a.k.a. Randonnet')
    print('==========================================')
    print('1. Flow Specification')
    print('==========================================')

    # Create the training data folder if they don't exist
    create_training_data_folder()

    # Load the segments
    # these variables are just 1D arrays
    seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, filename = load_segments()

    # Check if the polylines file exists
    if os.path.exists(os.path.join(PATH_PREFIX, 'data', 'flows', filename + '.polylines.pickle')):
        print(f'Loading polylines from {os.path.join(PATH_PREFIX, 'data', 'flows', filename + ".polylines.pickle")}')
        polylines = load_polylines_from_file(filename)
    else:
        # Allow the user to draw flows directly on the map
        command = '' 
        polylines = []
        while command != 'q':
            # If there are too many segments, we will sample a subset of them
            MAX_SEGMENTS = 2000
            if len(seg_from_lat) > MAX_SEGMENTS:
                print(f'Sampling {MAX_SEGMENTS} segments from the {len(seg_from_lat)} segments')
                np.random.seed(42)
                sampled_indices = np.random.choice(len(seg_from_lat), MAX_SEGMENTS, replace=False)
            else:
                sampled_indices = None
                
            polyline = flow_specification_interface(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, polylines, filter=sampled_indices)
            polylines.append(polyline)
            command = input("Press 'q' to quit, 'c' to continue: ")

        print(f"You drew {len(polylines)} polylines")
    
        # Write the polylines to a file
        write_polylines_to_file(polylines, filename) # polyline is a list of (lon, lat) tuples

    print('==========================================')
    print('2. Segment Annotation')
    print('==========================================')
    EPSILON = 0.5 # degrees latitude or longitude
    RESAMPLE = True
    N_RESAMPLE = 100

    # To store all the indices of the segments seen by the user
    admitted_indices = []
    # To store the indices of the segments labelled
    labelled_indices = []
    label = []
    # To store all the relegated segments by the user
    relegated_indices = []

    flow_id = 0
    

    for i_flow in range(len(polylines)):
        user_wants_to_proceed = True
        while user_wants_to_proceed: 
            print(f'Processing Flow {i_flow + 1}/{len(polylines)}')
            print('------------------------------------------')
            
            # Resample the polyline
            polylines_resampled = resample_polyline(polylines[i_flow], 0.5)
            # Obtain the epsilon-neighborhood segments
            neighbor_indices = get_epsilon_neighborhood(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, polylines_resampled, EPSILON)
            print(f'Found {len(neighbor_indices)} segments within {EPSILON} degrees of the polyline')
            neighbor_indices = get_long_enough_segments(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, 0.5, neighbor_indices)
            print(f'Found {len(neighbor_indices)} segments at least 0.5 degrees long')

            # Plot the segments
            # plot_segments_with_labels(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, filter=neighbor_indices)

            # This list maintains the indices of the segments that have not been admitted yet
            indices_pool = subtract_lists(neighbor_indices.tolist(), admitted_indices)
            indices_pool = np.array(indices_pool)

            # Resample the indices if the number of admitted segments is too large
            if len(indices_pool) > N_RESAMPLE and RESAMPLE:
                # Only keep N_RESAMPLE indices from the neighbor_indices
                np.random.seed(42)
                neighbor_indices = np.random.choice(indices_pool, N_RESAMPLE, replace=False)
                print(f'Resampled to {len(neighbor_indices)} indices')

            # ==============================
            # Base model TDF parameters 
            # Parameters for the similarity model   
            theta = {
                'psi_bar': TDFParams(tau=0.1, alpha=15., x0=1.),  # psi_bar = 1 - psi, where psi is the cosine similarity
                'wO': TDFParams(tau=0.1, alpha=15., x0=1.),  # weight of overlap
                'wH': TDFParams(tau=0.1, alpha=15., x0=1.),   # weight of horizontal separation
                'O': TDFParams(tau=0.1, alpha=15., x0=1.),   # overlap
                'H': TDFParams(tau=0.1, alpha=15., x0=1.)   # horizontal separation
            }

            # Handling the directionality of the flow: we compute the similarity matrix of all segments against two directions of the polyline
            for direction in [0, 1]:
                flow_id += 1
                print(f'Flow {i_flow + 1}/{len(polylines)}: direction {direction + 1}/2. Flow ID: {flow_id}')
                if direction == 0:
                    m_polyline = polylines[i_flow]
                else:
                    m_polyline = polylines[i_flow][::-1] # reverse the polyline
                m_polyline = np.array(m_polyline) # convert to numpy array
                similarity_matrix_polyline = get_similarity_matrix_polyline(seg_from_lat[neighbor_indices], seg_from_lon[neighbor_indices], seg_to_lat[neighbor_indices], seg_to_lon[neighbor_indices], m_polyline, theta)

                # Find the indices of the rows of similarity_matrix_polyline that are all zeros
                zero_rows_indices = np.where(np.all(similarity_matrix_polyline == 0, axis=1))[0]
                neighbor_indices_d = np.delete(neighbor_indices, zero_rows_indices) # the indices of the segments that are not aligned with the polyline's direction
                print(f'Admitted {len(neighbor_indices_d)} segments for this direction')

                # ==============================
                # Flow refinement
                print(f'Seeking user feedback on the flow... Total segments: {len(neighbor_indices_d)}. Press "s" to skip this flow, "z" to undo the last action.')

                # Open the flow refinement interface, relegated_segments is a subset of neighbor_indices_d, not counting from 0
                relegated_segments, again_flag = flow_refinement_interface(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, polylines[i_flow], neighbor_indices_d)
                
                if relegated_segments is None:
                    # This direction should be skipped
                    print(f'Skipping flow {i_flow + 1}/{len(polylines)}, direction {direction + 1}/2 due to user action')
                    continue

                print(f'Relegated {len(relegated_segments)} segments out of {len(neighbor_indices_d)}')
                print(f'Labelled {len(neighbor_indices_d) - len(relegated_segments)} segments')

                
                admitted_indices.extend(neighbor_indices_d)
                labelled_indices_for_this_flow = subtract_lists(neighbor_indices_d, relegated_segments) # relegated_segments is a subset of neighbor_indices_d, not counting from 0
                labelled_indices.extend(labelled_indices_for_this_flow)
                label.extend([flow_id] * len(labelled_indices_for_this_flow))
                relegated_indices.extend(relegated_segments)

                if not again_flag:
                    print('Will NOT sample another set of segments for this flow!')
                    user_wants_to_proceed = False 
                else:
                    print('Will sample another set of segments for this flow!')

        # Write the admitted segments to a file
        write_label_data_to_yaml_file(filename, admitted_indices, labelled_indices, label)

    print('Data Annotation Completed')
    print('==========================================') # end of flow labeling