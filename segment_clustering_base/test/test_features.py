import unittest
import pickle
# Add the parent directory to the Python path
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the parent of the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from path_prefix import PATH_PREFIX

from features import get_cosine_distance, get_overlap_distance, get_seg_from_to, get_horizontal_separation, get_segment_length

# Load the segments from the pickle file
# with open(PATH_PREFIX + "data/segments/flight_segments_1716508800.pickle", "rb") as f:
#     segments = pickle.load(f)

# seg_from_lat = segments['seg_from_lat']
# seg_from_lon = segments['seg_from_lon']
# seg_to_lat = segments['seg_to_lat']
# seg_to_lon = segments['seg_to_lon']



class TestFeatures(unittest.TestCase):
    def test_identical_segments(self):
        print('==========================================')
        print('Test 1: Identical segments')
        print('==========================================')
        # Just two simple segments that are identical
        seg_from_lat = np.array([0, 0]) # seg1, seg2
        seg_from_lon = np.array([0, 0]) # seg1, seg2
        seg_to_lat = np.array([0, 0]) # seg1, seg2
        seg_to_lon = np.array([10, 10]) # seg1, seg2

        # Convert to radians
        seg1_from, seg1_to, seg2_from, seg2_to = get_seg_from_to(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon)

        cosine_distance, flow_direction_similarity = get_cosine_distance(seg1_from, seg1_to, seg2_from, seg2_to)
        overlap = get_overlap_distance(seg1_from, seg1_to, seg2_from, seg2_to)
        
        horizontal_separation = get_horizontal_separation(seg1_from, seg1_to, seg2_from, seg2_to)

        print('Cosine distance:')
        print(cosine_distance)
        print('Flow direction similarity:')
        print(flow_direction_similarity)
        print('Overlap distance:')
        print(overlap)
        print('Horizontal separation:')
        print(horizontal_separation)

        self.assertAlmostEqual(cosine_distance, 1, delta=0.001)
        self.assertAlmostEqual(overlap, 0.5, delta=0.001) # 0.5 because they overlap completely
        self.assertAlmostEqual(horizontal_separation, 0, delta=0.001)
        
    def test_identical_segments_2(self):
        print('==========================================')
        print('Test 1.2: Identical segments')
        print('==========================================')
        # Just two simple segments that are identical
        seg_from_lat = np.array([0, 0]) # seg1, seg2
        seg_from_lon = np.array([0, 0]) # seg1, seg2
        seg_to_lat = np.array([10, 10]) # seg1, seg2
        seg_to_lon = np.array([0, 0]) # seg1, seg2

        # Convert to radians
        seg1_from, seg1_to, seg2_from, seg2_to = get_seg_from_to(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon)

        cosine_distance, flow_direction_similarity = get_cosine_distance(seg1_from, seg1_to, seg2_from, seg2_to)
        overlap = get_overlap_distance(seg1_from, seg1_to, seg2_from, seg2_to)
        
        horizontal_separation = get_horizontal_separation(seg1_from, seg1_to, seg2_from, seg2_to)

        print('Cosine distance:')
        print(cosine_distance)
        print('Flow direction similarity:')
        print(flow_direction_similarity)
        print('Overlap distance:')
        print(overlap)
        print('Horizontal separation:')
        print(horizontal_separation)

        self.assertAlmostEqual(cosine_distance, 1, delta=0.001)
        self.assertAlmostEqual(overlap, 0.5, delta=0.001) # 0.5 because they overlap completely
        self.assertAlmostEqual(horizontal_separation, 0, delta=0.001)
        
    def test_perpendicular_segments(self):
        print('==========================================')
        print('Test 2: Perpendicular segments')
        print('==========================================')
        # Just two simple segments that are perpendicular
        seg_from_lat = np.array([0, 0]) # seg1, seg2
        seg_from_lon = np.array([0, 0]) # seg1, seg2
        seg_to_lat = np.array([0, 1]) # seg1, seg2
        seg_to_lon = np.array([10, 0]) # seg1, seg2

        # Convert to radians
        seg1_from, seg1_to, seg2_from, seg2_to = get_seg_from_to(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon)

        cosine_distance, flow_direction_similarity = get_cosine_distance(seg1_from, seg1_to, seg2_from, seg2_to)

        print('Cosine distance:')
        print(cosine_distance)
        print('Flow direction similarity:')
        print(flow_direction_similarity)
        
        horizontal_separation = get_horizontal_separation(seg1_from, seg1_to, seg2_from, seg2_to)

        print('Horizontal separation:')
        print(horizontal_separation)
        
        print('Segment length:')
        print('Seg1:', get_segment_length(seg1_from, seg1_to))
        print('Seg2:', get_segment_length(seg2_from, seg2_to))

        self.assertAlmostEqual(cosine_distance, 0, delta=0.001)

    def test_overlap_segments(self):
        # Two segments that do not overlap
        seg_from_lat = np.array([0, 0]) # seg1, seg2
        seg_from_lon = np.array([0, 10]) # seg1, seg2
        seg_to_lat = np.array([0, 0]) # seg1, seg2
        seg_to_lon = np.array([10, 20]) # seg1, seg2

        # Convert to radians
        seg1_from, seg1_to, seg2_from, seg2_to = get_seg_from_to(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon)

        cosine_distance, flow_direction_similarity = get_cosine_distance(seg1_from, seg1_to, seg2_from, seg2_to)

        print('==========================================')
        print('Test 3: Overlap segments')
        print('==========================================')

        print('Cosine distance:')
        print(cosine_distance)
        print('Flow direction similarity:')
        print(flow_direction_similarity)
        
        overlap = get_overlap_distance(seg1_from, seg1_to, seg2_from, seg2_to)

        print('Overlap distance:')
        print(overlap)

        self.assertAlmostEqual(overlap, 1, delta=0.1)

    def test_overlap_segments_separated(self):
        # Two segments that do not overlap
        seg_from_lat = np.array([0, 0]) # seg1, seg2
        seg_from_lon = np.array([0, 20]) # seg1, seg2
        seg_to_lat = np.array([0, 0]) # seg1, seg2
        seg_to_lon = np.array([10, 30]) # seg1, seg2

        # Convert to radians
        seg1_from, seg1_to, seg2_from, seg2_to = get_seg_from_to(seg_from_lat, seg_from_lon, seg_to_lat, seg_to_lon)

        print('==========================================')
        print('Test 4: Overlap segments separated')
        print('==========================================')

        cosine_distance, flow_direction_similarity = get_cosine_distance(seg1_from, seg1_to, seg2_from, seg2_to)

        print('Cosine distance:')
        print(cosine_distance)
        print('Flow direction similarity:')
        print(flow_direction_similarity)

        overlap = get_overlap_distance(seg1_from, seg1_to, seg2_from, seg2_to)

        print('Overlap distance:')
        print(overlap)

        self.assertAlmostEqual(overlap, 1.5, delta=0.1)

if __name__ == '__main__':
    unittest.main()