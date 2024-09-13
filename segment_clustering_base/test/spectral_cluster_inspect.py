import pickle
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from sklearn.cluster import DBSCAN
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

import os
import sys

# Get the current directory
current_dir = os.path.dirname(os.path.abspath('__file__'))

# Get the segment_clustereuse directory
segment_clustereuse_dir = os.path.join(current_dir, 'segment_clustering_base')
print('segment_clustereuse_dir:', segment_clustereuse_dir)

# Add the parent directory to sys.path
sys.path.append(segment_clustereuse_dir)
sys.path.append(current_dir)

from segment_clustereuse.dsa import TDFParams
from segment_clustereuse.spectral_clustereuse import get_spectral_cluster, plot_segments_with_labels, get_spectral_embedding
from segment_clustereuse.sim_model import get_similarity_matrix, theta, explain_similarity
from path_prefix import PATH_PREFIX

# Load the segments from the pickle file
with open(PATH_PREFIX + '/data/segments/flight_segments_1716508800.pickle', 'rb') as f:
    segments_file = pickle.load(f)
    print('Segments file loaded')
    
seg_from_lat = segments_file['seg_from_lat']
seg_from_lon = segments_file['seg_from_lon']
seg_to_lat = segments_file['seg_to_lat']
seg_to_lon = segments_file['seg_to_lon']

# Convert deque to np.array
seg_from_lat = np.array(seg_from_lat)
seg_from_lon = np.array(seg_from_lon)
seg_to_lat = np.array(seg_to_lat)
seg_to_lon = np.array(seg_to_lon)

np.random.seed(42)
indices = np.random.choice(len(seg_from_lat), 200, replace=False).tolist()
seg_from_lat = seg_from_lat[indices]
seg_from_lon = seg_from_lon[indices]
seg_to_lat = seg_to_lat[indices]
seg_to_lon = seg_to_lon[indices]

print(f'Segments are sampled')

the = {
        'psi_bar': TDFParams(tau=0.01, alpha=12., x0=100.),  # psi_bar = 1 - psi, where psi is the cosine similarity
        'wO': TDFParams(tau=0.01, alpha=24., x0=1.),  # weight of overlap
        'wH': TDFParams(tau=0.01, alpha=24., x0=1.),   # weight of horizontal separation
        'O': TDFParams(tau=0.9, alpha=6., x0=100.),   # overlap
        'H': TDFParams(tau=0.005, alpha=60., x0=100.)   # horizontal separation
}

fig, ax = plt.subplots()

similarity_matrix = get_similarity_matrix(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, the)
embeddings = get_spectral_embedding(similarity_matrix)
scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1])

selected_indices = [] # for one polygon
selected_indicess = [] # for multiple polygons, each is selected_indices for one polygon

color_label = np.zeros(len(seg_from_lat)) - 1
color_index = -1

def onselect(verts):
    global selected_indices, selected_indicess, color_label, color_index
    color_index += 1
    path = Path(verts)
    selected_indices = np.where(path.contains_points(embeddings))[0].tolist()
    selected_indicess.append(selected_indices)
    color_label[selected_indices] = color_index
    print(f"Selected indices: {selected_indices}")

# Create the PolygonSelector
poly_selector = PolygonSelector(ax, onselect)

plt.title("Click to select points inside a polygon\nPress 'esc' to start a new polygon")
plt.show()

# Plot the selected segments
plot_segments_with_labels(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, cluster_labels=color_label, filter=None)
plt.show()

# Print the similarity matrix with selected indices
indices_flat = [index for sublist in selected_indicess for index in sublist]
print(f'Similarity matrix: {similarity_matrix[indices_flat]}')

print('OK')