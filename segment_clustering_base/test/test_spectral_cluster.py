import pickle
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from sklearn.cluster import DBSCAN

# Enable sampling to reduce the number of segments for testing
ENABLE_SAMPLING = True
SAMPLING_SEED = 42
N_SAMPLES = 300

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the parent of the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from segment_clustering_base.dsa import TDFParams
from segment_clustering_base.spectral_clustereuse import get_spectral_cluster, plot_segments_with_labels, get_spectral_embedding
from segment_clustering_base.sim_model import get_similarity_matrix, theta
from path_prefix import PATH_PREFIX

def save_similarity_matrix(similarity_matrix):
    # Create the similarity_matrix folder if it does not exist
    similarity_matrix_dir = os.path.join(PATH_PREFIX, 'data', 'similarity_matrix')
    if not os.path.exists(similarity_matrix_dir):
        os.makedirs(similarity_matrix_dir)
        
    with open(PATH_PREFIX + '/data/similarity_matrix/similarity_matrix.pickle', 'wb') as f:
        pickle.dump(similarity_matrix, f)

def load_similarity_matrix():
    # Create the path to the similarity matrix file
    similarity_matrix_path = os.path.join(PATH_PREFIX, 'data', 'similarity_matrix', 'similarity_matrix.pickle')
    
    # Check if the file exists
    if os.path.exists(similarity_matrix_path):
        with open(similarity_matrix_path, 'rb') as f:
            similarity_matrix = pickle.load(f)
        print('Similarity matrix loaded from file')
        return similarity_matrix
    else:
        print('No saved similarity matrix found')
        return None
    
def sample_segments(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon):
    np.random.seed(SAMPLING_SEED)
    indices = np.random.choice(len(seg_from_lat), N_SAMPLES, replace=False).tolist()
    seg_from_lat = seg_from_lat[indices]
    seg_from_lon = seg_from_lon[indices]
    seg_to_lat = seg_to_lat[indices]
    seg_to_lon = seg_to_lon[indices]
    
    print(f'{N_SAMPLES} segments are sampled')
    return seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon

if __name__ == '__main__':
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
    
    if ENABLE_SAMPLING: 
        seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon = sample_segments(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon)
    
    # Get the similarity matrix
    the = {
        'psi_bar': TDFParams(tau=0.01, alpha=15., x0=1.),  # psi_bar = 1 - psi, where psi is the cosine similarity
        'wO': TDFParams(tau=0.01, alpha=15., x0=1.),  # weight of overlap
        'wH': TDFParams(tau=0.01, alpha=15., x0=1.),   # weight of horizontal separation
        'O': TDFParams(tau=0, alpha=200., x0=1.),   # overlap
        'H': TDFParams(tau=0, alpha=200., x0=1.)   # horizontal separation
    }
    
    print('The following are the parameters used for the similarity model:')
    print(the)
    
    similarity_matrix = get_similarity_matrix(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, the)
    save_similarity_matrix(similarity_matrix)
    # similarity_matrix = load_similarity_matrix()
    
    # Get the embeddings
    embeddings = get_spectral_embedding(similarity_matrix)
    
    # Perform spectral embedding clustering using DBSCAN
    # Set the parameters for DBSCAN
    eps = 1e-3  # Maximum distance between two samples for them to be considered as in the same neighborhood
    min_samples = 4  # Minimum number of samples in a neighborhood for a point to be considered as a core point
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)
    
    print(f'Number of clusters: {len(set(cluster_labels))}')
    print(f'Number of noise points: {sum(cluster_labels == -1)}')
    
    # Get the spectral cluster
    # cluster_labels = get_spectral_cluster(similarity_matrix, n_clusters=10)
    
    # Scatter plot the embeddings
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar()
    plt.show()
    
    # Show the segments with the cluster labels
    plot_segments_with_labels(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, cluster_labels=cluster_labels)
    
    