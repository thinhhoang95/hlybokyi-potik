import numpy as np
from tqdm import tqdm
from dsa import TDFParams, tdf
from features import get_all_features

# Parameters for the similarity model
theta = {
    'psi_bar': TDFParams(tau=0.1, alpha=15., x0=1.),  # psi_bar = 1 - psi, where psi is the cosine similarity
    'wO': TDFParams(tau=0.1, alpha=15., x0=1.),  # weight of overlap
    'wH': TDFParams(tau=0.1, alpha=15., x0=1.),   # weight of horizontal separation
    'O': TDFParams(tau=0.1, alpha=15., x0=1.),   # overlap
    'H': TDFParams(tau=0.1, alpha=15., x0=1.)   # horizontal separation
}

def get_similarity_matrix(seg_from_lat: np.ndarray, seg_from_lon: np.ndarray, seg_to_lat: np.ndarray, seg_to_lon: np.ndarray, theta: dict) -> np.ndarray:
    # The similarity between two segments is given by the following formula:
    # similarity(i,j) = TDF(psi_bar) + TDF(psi_bar) * TDF(wO - 0.5) + TDF(psi_bar) * TDF(wH)
    
    n_segs = len(seg_from_lat)
    similarity_matrix = np.zeros((n_segs, n_segs))
    for i in tqdm(range(n_segs)):
        for j in range(n_segs):
            if i == j:
                similarity_matrix[i, j] = 1
            elif i > j:
                similarity_matrix[i, j] = similarity_matrix[j, i]
            else:
                seg1_from = np.radians([seg_from_lat[i], seg_from_lon[i]])
                seg1_to = np.radians([seg_to_lat[i], seg_to_lon[i]])
                seg2_from = np.radians([seg_from_lat[j], seg_from_lon[j]])
                seg2_to = np.radians([seg_to_lat[j], seg_to_lon[j]])
                # if segment 1 approximates zero, or segment 2 approximates zero, the similarity is 0
                if np.linalg.norm(seg1_from - seg1_to) < 1e-6 or np.linalg.norm(seg2_from - seg2_to) < 1e-6:
                    similarity_matrix[i, j] = 0
                    continue
                cosine_distance, flow_direction_similarity, overlap, horizontal_separation = get_all_features(seg1_from, seg1_to, seg2_from, seg2_to)
                if flow_direction_similarity < 0.5: # For perpendicular or opposite flow directions, the similarity is 0
                    similarity_matrix[i, j] = 0
                    # print(f'Segment {i} and segment {j} are perpendicular or opposite!')
                else:
                    psi_bar = 1 - cosine_distance
                    # Debugging: print the contribution of each term to the similarity matrix
                    contribs = {
                        'psi_bar': tdf(psi_bar, **theta['psi_bar']),
                        'wO': tdf(psi_bar, **theta['wO']),
                        'wH': tdf(psi_bar, **theta['wH']),
                        'O': tdf(overlap, **theta['O']),
                        'H': tdf(horizontal_separation, **theta['H'])
                    }
                    # Enable for debugging
                    # print(f'Segment {i} and segment {j}')
                    # print(contribs)
                    
                    # Divided by 3 to normalize the maximum value of the similarity matrix to 1
                    similarity_matrix[i, j] = (tdf(psi_bar, **theta['psi_bar']) \
                                            + tdf(psi_bar, **theta['wO']) * tdf(overlap, **theta['O']) \
                                            + tdf(psi_bar, **theta['wH']) * tdf(horizontal_separation, **theta['H']))/3.
                                            
    return similarity_matrix

def explain_similarity(seg_from_lat: np.ndarray, seg_from_lon: np.ndarray, seg_to_lat: np.ndarray, seg_to_lon: np.ndarray, theta: dict, i: int, j: int):
    seg1_from = np.radians([seg_from_lat[i], seg_from_lon[i]])
    seg1_to = np.radians([seg_to_lat[i], seg_to_lon[i]])
    seg2_from = np.radians([seg_from_lat[j], seg_from_lon[j]])
    seg2_to = np.radians([seg_to_lat[j], seg_to_lon[j]])
    
    print(f'Segment 1: {seg1_from} to {seg1_to}')
    print(f'Segment 2: {seg2_from} to {seg2_to}')
    
    # if segment 1 approximates zero, or segment 2 approximates zero, the similarity is 0
    if np.linalg.norm(seg1_from - seg1_to) < 1e-6 or np.linalg.norm(seg2_from - seg2_to) < 1e-6:
        print('Returns 0 because one of the segment is too short!')
        print(f'Norm of segment 1: {np.linalg.norm(seg1_from - seg1_to)}')
        print(f'Norm of segment 2: {np.linalg.norm(seg2_from - seg2_to)}')
        return
    
    cosine_distance, flow_direction_similarity, overlap, horizontal_separation = get_all_features(seg1_from, seg1_to, seg2_from, seg2_to)

    if flow_direction_similarity < 0.5:
        print('Returns 0 because the flow directions are perpendicular or opposite!')
        print(f'Flow direction similarity: {flow_direction_similarity}')
        return
    
    psi_bar = 1 - cosine_distance
    contribs = {
        'tdf(psi_bar)': tdf(psi_bar, **theta['psi_bar']),
        'tdf(wO)': tdf(psi_bar, **theta['wO']),
        'tdf(O)': tdf(overlap, **theta['O']),
        'O': overlap,
        'tdf(wH)': tdf(psi_bar, **theta['wH']),
        'H': horizontal_separation,
        'tdf(H)': tdf(horizontal_separation, **theta['H'])
    }
    
    print('==========')
    print('psi_bar: ', tdf(psi_bar, **theta['psi_bar']))
    print('O_term: ', tdf(psi_bar, **theta['wO']) * tdf(overlap, **theta['O']))
    print('H_term: ', tdf(psi_bar, **theta['wH']) * tdf(horizontal_separation, **theta['H']))
    print('==========')
    for key in contribs:
        print(f'{key}: {contribs[key]}')
    print('==========')

    similarity = (tdf(psi_bar, **theta['psi_bar']) \
                + tdf(psi_bar, **theta['wO']) * tdf(overlap, **theta['O']) \
                + tdf(psi_bar, **theta['wH']) * tdf(horizontal_separation, **theta['H']))/3.
    
    print(f'Similarity between segment {i} and segment {j} is {similarity}')
    print('==========')

from matplotlib import pyplot as plt

def inspect_direction_between_two_segments(seg1_from: np.ndarray, seg1_to: np.ndarray, seg2_from: np.ndarray, seg2_to: np.ndarray):
    cosine_distance, flow_direction_similarity, overlap, horizontal_separation = get_all_features(seg1_from, seg1_to, seg2_from, seg2_to)
    print(f'Cosine distance: {cosine_distance}')
    print(f'Flow direction similarity: {flow_direction_similarity}')
    print(f'Overlap: {overlap}')
    print(f'Horizontal separation: {horizontal_separation}')

    plt.figure()
    plt.arrow(seg1_from[0], seg1_from[1], seg1_to[0]-seg1_from[0], seg1_to[1]-seg1_from[1], 
              head_width=0.5, head_length=0.5, fc='r', ec='r', label='Segment')
    plt.arrow(seg2_from[0], seg2_from[1], seg2_to[0]-seg2_from[0], seg2_to[1]-seg2_from[1], 
              head_width=0.5, head_length=0.5, fc='b', ec='b', label='Polyline')
    plt.legend()
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Segment Directions (NOT LAT/LON)')
    plt.axis('equal')
    plt.show()

def get_similarity_matrix_polyline(seg_from_lat: np.ndarray, seg_from_lon: np.ndarray, seg_to_lat: np.ndarray, seg_to_lon: np.ndarray, polyline: np.ndarray, theta: dict) -> np.ndarray:
    """
    Compute the similarity matrix between segments and a polyline.

    Args:
        seg_from_lat (1D np.ndarray): the from latitude of the segments
        seg_from_lon (1D np.ndarray): the from longitude of the segments
        seg_to_lat (1D np.ndarray): the to latitude of the segments
        seg_to_lon (1D np.ndarray): the to longitude of the segments
        polyline (2D np.ndarray): the polyline, with shape (n_points, 2)
        theta (dict): the parameters for the similarity model
        
    Returns:
        np.ndarray: the similarity matrix
    """

    similarity_matrix = np.zeros((len(seg_from_lat), len(polyline) - 1)) # row: segments, col: polyline
    for i in tqdm(range(len(seg_from_lat))): # for each segment
        for j in range(len(polyline) - 1): # for each segment in the polyline

            seg1_from = np.radians([seg_from_lat[i], seg_from_lon[i]])
            seg1_to = np.radians([seg_to_lat[i], seg_to_lon[i]])
            polyline_from = np.radians([polyline[j,1], polyline[j,0]])
            polyline_to = np.radians([polyline[j+1,1], polyline[j+1,0]])

            # print(f'Segment {i}: {np.degrees(seg1_from)} to {np.degrees(seg1_to)}')
            # print(f'Polyline {j}: {np.degrees(polyline_from)} to {np.degrees(polyline_to)}')

            # print(f'Dot product of segment {i} and polyline {j}: {np.dot(seg1_to - seg1_from, polyline_to - polyline_from)}')

            # if segment 1 approximates zero, the similarity is 0
            if np.linalg.norm(seg1_from - seg1_to) < 1e-6:
                similarity_matrix[i, j] = 0
                continue
            cosine_distance, flow_direction_similarity, overlap, horizontal_separation = get_all_features(seg1_from, seg1_to, polyline_from, polyline_to)
            if flow_direction_similarity < 0.5: # For perpendicular or opposite flow directions, the similarity is 0
                similarity_matrix[i, j] = 0
                # inspect_direction_between_two_segments(seg1_from, seg1_to, polyline_from, polyline_to)
                # print(f'Segment {i} and polyline {j} are perpendicular or opposite!')
            else:
                psi_bar = 1 - cosine_distance
                # Debugging: print the contribution of each term to the similarity matrix
                contribs = {
                    'psi_bar': tdf(psi_bar, **theta['psi_bar']),
                    'wO': tdf(psi_bar, **theta['wO']),
                    'wH': tdf(psi_bar, **theta['wH']),
                    'O': tdf(overlap, **theta['O']),
                    'H': tdf(horizontal_separation, **theta['H'])
                }
                # Enable for debugging
                # print(f'Segment {i} and segment {j}')
                # print(contribs)
                
                # Divided by 3 to normalize the maximum value of the similarity matrix to 1
                similarity_matrix[i, j] = (tdf(psi_bar, **theta['psi_bar']) \
                                            + tdf(psi_bar, **theta['wO']) * tdf(overlap, **theta['O']) \
                                            + tdf(psi_bar, **theta['wH']) * tdf(horizontal_separation, **theta['H']))/3.
                
    return similarity_matrix
