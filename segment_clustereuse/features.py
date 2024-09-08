import numpy as np
from scipy.spatial import distance

def get_seg_from_to(lat_from, lon_from, lat_to, lon_to):
    """Convert latitude and longitude to radians. Also reorganize the coordinates by segment.

    Args:
        lat_from (np.array (2,)): latitude of the from points (lat1, lat2 for seg1 and seg2 respectively)
        lon_from (np.array (2,)): longitude of the from points (lon1, lon2 for seg1 and seg2 respectively)
        lat_to (np.array (2,)): latitude of the to points (lat1, lat2 for seg1 and seg2 respectively)
        lon_to (np.array (2,)): longitude of the to points (lon1, lon2 for seg1 and seg2 respectively)

    Returns:
        seg1_from, seg1_to, seg2_from, seg2_to (np.array (2,)): the from and to points in radians
    """
    seg1_from = np.radians([lat_from[0], lon_from[0]])
    seg1_to = np.radians([lat_to[0], lon_to[0]])
    seg2_from = np.radians([lat_from[1], lon_from[1]])
    seg2_to = np.radians([lat_to[1], lon_to[1]])
    return seg1_from, seg1_to, seg2_from, seg2_to


def to_cartesian(lat, lon):
    """Convert latitude and longitude to Cartesian coordinates.

    Args:
        lat (float): latitude
        lon (float): longitude

    Returns:
        np.array: the Cartesian coordinates
    """
    return np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])

def get_cosine_distance(seg1_from, seg1_to, seg2_from, seg2_to):
    """Get the cosine distance between two segments. Use get_seg_from_to to convert the segments to radians.
    The cosine distance is the cosine of the angle between the two great circle planes going through each segment.
    The range of the cosine distance is between 0 and 1. 1 means the two segments are parallel, and 0 means the two segments are orthogonal.

    Args:
        seg1_from (np.array): the from point of the first segment (lat, lon)
        seg1_to (np.array): the to point of the first segment (lat, lon)
        seg2_from (np.array): the from point of the second segment (lat, lon)
        seg2_to (np.array): the to point of the second segment (lat, lon)

    Returns:
        np.array: the psi matrix
    """
    # Prompt: give the segments (in geographical coordinates) for seg1 and seg2, we compute the feature: the cosine of angle between two great circle planes going through these two segments. We do that by finding the normal vectors of the great circle planes going through each segment, then we perform a dot product. The result is put in a matrix psi, where psi[i,j]=psi[j,i] the cosine of the angles. Thanks.
    # Convert latitude and longitude to radians
    # seg1_from, seg1_to = get_seg_from_to(seg1_from_lat, seg1_from_lon, seg1_to_lat, seg1_to_lon)
    # seg2_from, seg2_to = get_seg_from_to(seg2_from_lat, seg2_from_lon, seg2_to_lat, seg2_to_lon)

    # Calculate normal vectors for each great circle plane
    normal1 = np.cross(
        [np.cos(seg1_from[0]) * np.cos(seg1_from[1]),
         np.cos(seg1_from[0]) * np.sin(seg1_from[1]),
         np.sin(seg1_from[0])],
        [np.cos(seg1_to[0]) * np.cos(seg1_to[1]),
         np.cos(seg1_to[0]) * np.sin(seg1_to[1]),
         np.sin(seg1_to[0])]
    )
    normal2 = np.cross(
        [np.cos(seg2_from[0]) * np.cos(seg2_from[1]),
         np.cos(seg2_from[0]) * np.sin(seg2_from[1]),
         np.sin(seg2_from[0])],
        [np.cos(seg2_to[0]) * np.cos(seg2_to[1]),
         np.cos(seg2_to[0]) * np.sin(seg2_to[1]),
         np.sin(seg2_to[0])]
    )

    # Normalize the vectors
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)
    
    flow_direction_similarity = np.sign(np.dot(normal1, normal2))

    # Calculate the cosine of the angle between the planes
    return np.abs(np.dot(normal1, normal2)), flow_direction_similarity # cos_angle between 0 and 1, flow_direction_similarity: 1 if parallel, -1 if antiparallel
    
def get_overlap_distance(seg1_from, seg1_to, seg2_from, seg2_to):
    """Get the overlap distance between two segments. Use get_seg_from_to to convert the segments to radians.

    Args:
        seg1_from (np.array): the from point of the first segment (lat, lon)
        seg1_to (np.array): the to point of the first segment (lat, lon)
        seg2_from (np.array): the from point of the second segment (lat, lon)
        seg2_to (np.array): the to point of the second segment (lat, lon)

    Returns:
        float: the overlap distance
    """
    # Convert latitude and longitude to radians
    # seg1_from, seg1_to = get_seg_from_to(seg1_from_lat, seg1_from_lon, seg1_to_lat, seg1_to_lon)
    # seg2_from, seg2_to = get_seg_from_to(seg2_from_lat, seg2_from_lon, seg2_to_lat, seg2_to_lon)

    seg1_from_cart = to_cartesian(*seg1_from)
    seg1_to_cart = to_cartesian(*seg1_to)
    seg2_from_cart = to_cartesian(*seg2_from)
    seg2_to_cart = to_cartesian(*seg2_to)

    # Calculate segment vectors
    seg1_vec = seg1_to_cart - seg1_from_cart
    seg2_vec = seg2_to_cart - seg2_from_cart

    # Find the pair of longest endpoints
    endpoints = np.array([seg1_from_cart, seg1_to_cart, seg2_from_cart, seg2_to_cart])
    
    # Calculate pairwise distances using broadcasting
    distances = np.linalg.norm(endpoints[:, np.newaxis] - endpoints, axis=2)
    
    # Set diagonal to -1 to exclude self-distances
    np.fill_diagonal(distances, -1)
    
    # Find the indices of the maximum distance
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    
    # Get the longest pair
    longest_pair = (endpoints[i], endpoints[j])

    # Calculate the vector between the longest pair
    v_vec = longest_pair[1] - longest_pair[0]

    # Project v onto seg1
    proj_v_on_seg1 = np.dot(v_vec, seg1_vec) / np.dot(seg1_vec, seg1_vec) * seg1_vec

    # Calculate lengths
    len_proj_v = np.linalg.norm(proj_v_on_seg1)
    len_seg1 = np.linalg.norm(seg1_vec)
    len_seg2 = np.linalg.norm(seg2_vec)

    # Calculate the overlap ratio
    overlap_ratio = len_proj_v / (len_seg1 + len_seg2)

    return overlap_ratio

# Function to calculate perpendicular distance from a point to a line segment
def point_to_segment_distance(point, segment_start, segment_end):
    """Calculate the perpendicular distance from a point to a line segment.

    Args:
        point (np.array): the point
        segment_start (np.array): the start point of the segment
        segment_end (np.array): the end point of the segment

    Returns:
        float: the perpendicular distance
    """
    segment_vec = segment_end - segment_start
    t = np.dot(point - segment_start, segment_vec) / np.dot(segment_vec, segment_vec)
    t = np.clip(t, 0, 1)
    projection = segment_start + t * segment_vec
    return distance.euclidean(point, projection)

def get_horizontal_separation(seg1_from, seg1_to, seg2_from, seg2_to):
    """Get the maximum horizontal separation between two segments. Use get_seg_from_to to convert the segments to radians.

    Args:
        seg1_from (np.array): the from point of the first segment (lat, lon)
        seg1_to (np.array): the to point of the first segment (lat, lon)
        seg2_from (np.array): the from point of the second segment (lat, lon)
        seg2_to (np.array): the to point of the second segment (lat, lon)

    Returns:
        float: the maximum horizontal separation
    """
    # Convert endpoints to Cartesian coordinates
    seg1_from_cart = to_cartesian(*seg1_from)
    seg1_to_cart = to_cartesian(*seg1_to)
    seg2_from_cart = to_cartesian(*seg2_from)
    seg2_to_cart = to_cartesian(*seg2_to)

    # # Calculate segment vectors
    # seg1_vec = seg1_to_cart - seg1_from_cart
    # seg2_vec = seg2_to_cart - seg2_from_cart

    # # Normalize segment vectors
    # seg1_unit = seg1_vec / np.linalg.norm(seg1_vec)
    # seg2_unit = seg2_vec / np.linalg.norm(seg2_vec)

    # Calculate perpendicular distances
    distances = [
        point_to_segment_distance(seg1_from_cart, seg2_from_cart, seg2_to_cart),
        point_to_segment_distance(seg1_to_cart, seg2_from_cart, seg2_to_cart),
        point_to_segment_distance(seg2_from_cart, seg1_from_cart, seg1_to_cart),
        point_to_segment_distance(seg2_to_cart, seg1_from_cart, seg1_to_cart)
    ]

    # Return the maximum separation
    return max(distances)

def get_segment_length(seg_from, seg_to):
    """Get the length of a segment. Use get_seg_from_to to convert the segment to radians.

    Args:
        seg_from (np.array): the from point of the segment (lat, lon)
        seg_to (np.array): the to point of the segment (lat, lon)

    Returns:
        float: the length of the segment
    """
    seg_from_cart = to_cartesian(*seg_from)
    seg_to_cart = to_cartesian(*seg_to)
    return np.linalg.norm(seg_to_cart - seg_from_cart)

def get_flow_direction_similarity(seg1_from, seg1_to, seg2_from, seg2_to):
    """Get the flow direction similarity between two segments. Use get_seg_from_to to convert the segments to radians.

    Args:
        seg1_from (np.array): the from point of the first segment (lat, lon)
        seg1_to (np.array): the to point of the first segment (lat, lon)
        seg2_from (np.array): the from point of the second segment (lat, lon)
        seg2_to (np.array): the to point of the second segment (lat, lon)
    """
    # Calculate the flow direction of each segment
    seg1_flow_dir = seg1_to - seg1_from
    seg2_flow_dir = seg2_to - seg2_from

    # Calculate the cosine similarity between the two flow directions
    return np.dot(seg1_flow_dir, seg2_flow_dir) / (np.linalg.norm(seg1_flow_dir) * np.linalg.norm(seg2_flow_dir))

def get_all_features(seg1_from, seg1_to, seg2_from, seg2_to):
    """Get all the features between two segments. Use get_seg_from_to to convert the segments to radians.

    Args:
        seg1_from (np.array): the from point of the first segment (lat, lon)
        seg1_to (np.array): the to point of the first segment (lat, lon)
        seg2_from (np.array): the from point of the second segment (lat, lon)
        seg2_to (np.array): the to point of the second segment (lat, lon)
    """
    cosine_distance, flow_direction_similarity = get_cosine_distance(seg1_from, seg1_to, seg2_from, seg2_to)
    overlap = get_overlap_distance(seg1_from, seg1_to, seg2_from, seg2_to)
    horizontal_separation = get_horizontal_separation(seg1_from, seg1_to, seg2_from, seg2_to)
    return cosine_distance, flow_direction_similarity, overlap, horizontal_separation