import os
import sys

import numpy as np
import yaml
from data_labeling import load_segments, load_polylines_from_file

# Add the parent directory of this python file to the search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the segment_clustering_base folder to the search path
sys.path.append(os.path.join(parent_dir, 'segment_clustering_base'))
sys.path.append(parent_dir)

from path_prefix import PATH_PREFIX

def read_label_yaml(filename: str):
    with open(os.path.join(PATH_PREFIX, 'data', 'c1_train', 'flows', filename + '.yaml'), 'r') as f:
        labels = yaml.load(f, Loader=yaml.FullLoader)
        # Create the numpy arrays: 
        admitted_indices = np.array(labels['admitted_indices'])
        labls = np.array(labels['label'])
        labelled_indices = np.array(labels['labelled_indices'])
    return admitted_indices, labls, labelled_indices

if __name__ == '__main__':
    # Load the segments
    seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, filename = load_segments()
    # Check if the polylines file exists
    if os.path.exists(os.path.join(PATH_PREFIX, 'data', 'flows', filename + '.polylines.pickle')):
        print(f'Loading polylines from {os.path.join(PATH_PREFIX, 'data', 'flows', filename + ".polylines.pickle")}')
        polylines = load_polylines_from_file(filename)
        print(f'Loaded {len(polylines)} polylines')
    else:
        print(f'Polylines file {os.path.join(PATH_PREFIX, 'data', 'flows', filename + ".polylines.pickle")} does not exist')
        exit(1)

    # Load the labels
    admitted_indices, labls, labelled_indices = read_label_yaml(filename)

    from spectral_clustereuse import plot_segments_with_labels # type:ignore
    
    for i in range(len(polylines)):
        m_polyline = np.array(polylines[i])
        labls_i = np.where(labls == (i * 2 + 1))[0]
        labelled_indices_i = labelled_indices[labls_i]
        plot_segments_with_labels(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon, filter=labelled_indices_i, show_labels=True, polyline=m_polyline)