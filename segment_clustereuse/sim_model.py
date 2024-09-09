import numpy as np
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
    for i in range(n_segs):
        for j in range(n_segs):
            if i == j:
                similarity_matrix[i, j] = 1
            elif i > j:
                similarity_matrix[i, j] = similarity_matrix[j, i]
            else:
                seg1_from = np.array([seg_from_lat[i], seg_from_lon[i]])
                seg1_to = np.array([seg_to_lat[i], seg_to_lon[i]])
                seg2_from = np.array([seg_from_lat[j], seg_from_lon[j]])
                seg2_to = np.array([seg_to_lat[j], seg_to_lon[j]])
                cosine_distance, flow_direction_similarity, overlap, horizontal_separation = get_all_features(seg1_from, seg1_to, seg2_from, seg2_to)
                if flow_direction_similarity < 0.5: # For perpendicular or opposite flow directions, the similarity is 0
                    similarity_matrix[i, j] = 0
                else:
                    psi_bar = 1 - cosine_distance
                    # Debugging: print the contribution of each term to the similarity matrix
                    contribs = {
                        'psi_bar': tdf(psi_bar, **theta['psi_bar']),
                        'wO': tdf(psi_bar, **theta['wO']),
                        'wH': tdf(psi_bar, **theta['wH']),
                        'O': tdf(overlap - 0.5, **theta['O']),
                        'H': tdf(horizontal_separation, **theta['H'])
                    }
                    print(f'Segment {i} and segment {j}')
                    print(contribs)
                    
                    # Divided by 3 to normalize the maximum value of the similarity matrix to 1
                    similarity_matrix[i, j] = (tdf(psi_bar, **theta['psi_bar']) \
                                            + tdf(psi_bar, **theta['wO']) * tdf(overlap - 0.5, **theta['O']) \
                                            + tdf(psi_bar, **theta['wH']) * tdf(horizontal_separation, **theta['H']))/3.
                                            
    return similarity_matrix