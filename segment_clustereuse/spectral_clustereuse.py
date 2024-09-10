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
        n_components=2, # latent dimensions
        random_state=random_state
    )

    return embedding

def get_spectral_cluster(similarity_matrix, n_clusters=2, random_state=42):
    """
    Perform spectral clustering on a given similarity matrix.

    Args:
        similarity_matrix (np.ndarray): A square matrix of shape (n_samples, n_samples) 
                                        representing the similarity between samples.
        n_clusters (int): The number of clusters to form. Default is 2.
        random_state (int): Determines random number generation for centroid initialization. Default is 42.

    Returns:
        np.ndarray: An array of cluster labels for each sample.
    """
    # Ensure the similarity matrix is symmetric
    if not np.allclose(similarity_matrix, similarity_matrix.T):
        raise ValueError("The similarity matrix must be symmetric")

    # Perform spectral clustering
    spectral_clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state
    )
    
    cluster_labels = spectral_clusterer.fit_predict(similarity_matrix)

    return cluster_labels

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_segments_with_labels(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, cluster_labels=None, filter=None, show_labels=False):
    # Create a new figure and axis with a map projection
    fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.set_aspect('auto')  # This allows the aspect ratio to adjust naturally

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Set the extent of the map to cover Europe
    ax.set_extent([-10, 40, 28, 75], crs=ccrs.PlateCarree())

    # Create a colormap
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # Plot the segments
    i: int = 0
    if cluster_labels is not None:  
        for from_lat, from_lon, to_lat, to_lon, label in zip(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, cluster_labels):
            if filter is not None and i not in filter:
                i += 1
                continue
            i += 1
            color = colors[np.where(unique_labels == label)[0][0]]
            ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                    color=color, linewidth=1., alpha=0.5, 
                    transform=ccrs.Geodetic())
            # Add a text label at the midpoint of the segment
            if show_labels:
                mid_lat = (from_lat + to_lat) / 2
                mid_lon = (from_lon + to_lon) / 2
                label = np.where((seg_from_lat == from_lat) & (seg_from_lon == from_lon) & (seg_to_lat == to_lat) & (seg_to_lon == to_lon))[0][0]
                ax.text(mid_lon, mid_lat, str(label), transform=ccrs.Geodetic(), fontsize=8, ha='center', va='center', color='black')
            # Add a marker x at the end of the segment
            ax.plot([to_lon], [to_lat], marker='x', color=color, transform=ccrs.Geodetic())
        # Show colorbar for the cluster labels
        # Create a ScalarMappable object for the colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=min(unique_labels), vmax=max(unique_labels)))
        sm.set_array([])  # This line is necessary for the colorbar to work correctly
        
        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label('Cluster Labels')
        
        # Set integer ticks on the colorbar
        cbar.set_ticks(unique_labels)
        cbar.set_ticklabels(unique_labels)
    else: # no color since no cluster labels specified / no cluster_label
        i: int = 0
        for from_lat, from_lon, to_lat, to_lon in zip(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon):
            if filter is not None and i not in filter:
                i += 1
                continue
            i += 1
            ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                    color='red', linewidth=1., alpha=0.5, 
                    transform=ccrs.Geodetic())
            # Add a text label at the midpoint of the segment
            if show_labels: 
                mid_lat = (from_lat + to_lat) / 2
                mid_lon = (from_lon + to_lon) / 2
                # label is the index of the segment
                label = np.where((seg_from_lat == from_lat) & (seg_from_lon == from_lon) & (seg_to_lat == to_lat) & (seg_to_lon == to_lon))[0][0]
                ax.text(mid_lon, mid_lat, str(label), transform=ccrs.Geodetic(), fontsize=8, ha='center', va='center', color='black')
            # Add a marker x at the end of the segment
            ax.plot([to_lon], [to_lat], marker='x', color='red', transform=ccrs.Geodetic())

    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    plt.show()