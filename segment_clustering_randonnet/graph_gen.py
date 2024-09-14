from turtle import pd
from typing import List, Tuple
import numpy as np
from scipy.sparse.csgraph import connected_components

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from path_prefix import PATH_PREFIX
from segment_clustering_base.sim_model import get_similarity_matrix


def create_adj_matrix(similarity_matrix: np.ndarray, threshold: float) -> float:
    return (similarity_matrix > threshold).astype(int)

def solve_base_graph_for_threshold(similarity_matrix: np.ndarray) -> Tuple[float, float]:
    def is_strongly_connected(adj_matrix: np.ndarray) -> bool:
        n_components, _ = connected_components(adj_matrix, directed=True, connection='strong')
        return n_components == 1

    unique_values = np.unique(similarity_matrix)
    left, right = 0, len(unique_values) - 1

    while left <= right:
        mid = (left + right) // 2
        threshold = unique_values[mid]
        adj_matrix = create_adj_matrix(similarity_matrix, threshold)

        if is_strongly_connected(adj_matrix):
            left = mid + 1
        else:
            right = mid - 1

    threshold = unique_values[right]

    # Return the base graph, which is the adjacency matrix
    return threshold, unique_values[len(unique_values) - 1]

def generate_base_graphs(similarity_matrix: np.ndarray, max_percentage: float = 0.75):
    threshold, max_threshold = solve_base_graph_for_threshold(similarity_matrix)
    adj_matrices = []
    # We attempt to generate several base graphs by thresholding the similarity matrix
    # at several different percentages around the threshold
    for percentage in np.linspace(0., max_percentage, 10): # 10 base graphs
        threshold = threshold + (max_threshold - threshold) * percentage
        adj_matrix = create_adj_matrix(similarity_matrix, threshold)
        adj_matrices.append(adj_matrix)

    # Remove duplicate base graphs
    adj_matrices = np.unique(adj_matrices, axis=0)

    return np.array(adj_matrices)

def get_flight_segments_prefix_of_hour():
    flow_dir = os.path.join(PATH_PREFIX, 'data', 'c1_train', 'flows')
    # List all csv files in the flow directory
    csv_files = [f for f in os.listdir(flow_dir) if f.endswith('.csv')]
    print(f'Found {len(csv_files)} CSV files in {flow_dir}')

    # Extract unique file prefixes
    file_prefixes = set()
    for file in csv_files:
        # Split the filename and take the first two parts
        parts = file.split('_')
        if len(parts) >= 3:
            prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"
            file_prefixes.add(prefix)

    file_prefixes = list(file_prefixes)
    return file_prefixes

def get_clean_filename(filepath: str) -> str:
    """
    Returns a clean filename without directory path or file extensions.

    Args:
        filepath (str): The full path of the file or just the filename.

    Returns:
        str: The clean filename without directory or extensions.
    """
    # Get the base filename without directory
    filename = os.path.basename(filepath)
    
    # Split the filename and keep only the part before the first dot
    clean_filename = filename.split('.')[0]
    
    return clean_filename
    

def get_training_data_for_prefix(prefix: str):
    flow_dir = os.path.join(PATH_PREFIX, 'data', 'c1_train', 'flows')
    flow_csv = [f for f in os.listdir(flow_dir) if f.startswith(prefix)]
    
    segment_dir = os.path.join(PATH_PREFIX, 'data', 'c1_train', 'segments')
    segment_csv = [f for f in os.listdir(segment_dir) if f.startswith(prefix)]

    for flow_file in flow_csv:
        # filename is without .csv
        filename_clean = get_clean_filename(flow_file)
        # Get the corresponding segment file
        segment_file = [f for f in segment_csv if f.startswith(filename_clean)]
        if len(segment_file) == 0:
            raise ValueError(f'No segment file found for {flow_file}')
        elif len(segment_file) > 1:
            raise ValueError(f'Ambiguous segment files found for {flow_file}: {segment_file}')
        else:
            segment_file = segment_file[0]

        # Load the segments
        df_segment = pd.read_csv(os.path.join(segment_dir, segment_file))
        from_lat = df_segment['from_lat'].values
        from_lng = df_segment['from_lon'].values
        to_lat = df_segment['to_lat'].values
        to_lng = df_segment['to_lon'].values

        # Compute similarity matrix for the segments
        similarity_matrix = compute_similarity_matrix(from_lat, from_lng, to_lat, to_lng)
        base_graphs_for_flow = generate_base_graphs(similarity_matrix)
        
        # base_graphs_for_flow is a 3D array of shape (n_base_graphs, n_segments, n_segments)






if __name__ == '__main__':
    similarity_matrix = np.array([
        [1, 0.5, 0.2],
        [0.5, 1, 0.3],
        [0.2, 0.3, 1]
    ])
    base_graphs = generate_base_graphs(similarity_matrix)
    print(f'Generated {base_graphs.shape[0]} base graphs')

    for i, base_graph in enumerate(base_graphs):
        print(f'Base graph {i+1}:')
        print(base_graph)
        print()

    # Test the prepare_folder function
    folder_path = os.path.join(PATH_PREFIX, 'data', 'c1_train', 'flows')
    unique_prefixes = prepare_folder(folder_path)
    print(f'Found {len(unique_prefixes)} unique file prefixes:')
    for prefix in unique_prefixes:
        print(prefix)