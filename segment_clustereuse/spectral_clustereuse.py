import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding

def get_spectral_embedding(similarity_matrix, random_state=42):
    """
    Perform spectral clustering on a given similarity matrix.

    Args:
        similarity_matrix (np.ndarray): A square matrix of shape (n_samples, n_samples) 
                                        representing the similarity between samples.
        n_clusters (int): The number of clusters to form. Default is 2.
        random_state (int): Determines random number generation for centroid initialization. Default is 42.

    Returns:
        tuple: A tuple containing the spectral embedding and the cluster labels.
    """
    # Ensure the similarity matrix is symmetric
    if not np.allclose(similarity_matrix, similarity_matrix.T):
        raise ValueError("The similarity matrix must be symmetric")

    # Compute the spectral embedding
    embedding = spectral_embedding(
        similarity_matrix,
        n_components=2,
        random_state=random_state
    )

    return embedding
