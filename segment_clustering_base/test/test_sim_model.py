import numpy as np
import matplotlib.pyplot as plt
# Import the parent directory to the current file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_model import get_similarity_matrix, theta
from dsa import TDFParams

def plot_segments(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon):
    fig, ax = plt.subplots()
    for i in range(len(seg_from_lat)):
        ax.plot([seg_from_lat[i], seg_to_lat[i]], [seg_from_lon[i], seg_to_lon[i]], label=f'{i}')
    ax.legend()
    plt.show()

def test_sim_model1():
    # How to read: each segment is one column. The first two rows are one end-point, the remaining two rows are the other end-point.
    seg_from_lat = np.array([0, 0.2, 0, 0])
    seg_from_lon = np.array([0, 0, 0, 0])
    seg_to_lat = np.array([0, 0, 1, -1])
    seg_to_lon = np.array([1, 1.5, 0, 0])
    
    plot_segments(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon)
    
    # Copy theta to a new variable called m_theta
    m_theta = theta.copy()
    
    sim_matrix = get_similarity_matrix(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, m_theta)
    print(sim_matrix)
    
if __name__ == "__main__":
    test_sim_model1()
