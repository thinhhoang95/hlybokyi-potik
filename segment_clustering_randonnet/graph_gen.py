from typing import List
import numpy as np
from scipy.sparse.csgraph import connected_components

def solve_base_graph(similarity_matrix: np.ndarray) -> np.ndarray:
    def is_strongly_connected(adj_matrix: np.ndarray) -> bool:
        n_components, _ = connected_components(adj_matrix, directed=True, connection='strong')
        return n_components == 1

    def create_adj_matrix(threshold: float) -> np.ndarray:
        return (similarity_matrix > threshold).astype(int)

    unique_values = np.unique(similarity_matrix)
    left, right = 0, len(unique_values) - 1

    while left <= right:
        mid = (left + right) // 2
        threshold = unique_values[mid]
        adj_matrix = create_adj_matrix(threshold)

        if is_strongly_connected(adj_matrix):
            left = mid + 1
        else:
            right = mid - 1

    threshold = unique_values[right]

    # Return the base graph, which is the adjacency matrix
    return adj_matrix

if __name__ == '__main__':
    similarity_matrix = np.array([
        [1, 0.5, 0.2],
        [0.5, 1, 0.3],
        [0.2, 0.3, 1]
    ])
    base_graph = solve_base_graph(similarity_matrix)
    print(base_graph)
