import numpy as np
from scipy.optimize import fsolve
import warnings
import math

# Earth's radius
Re = 6371e3 # m

def latlon2xyz(lat, lon):
    """Converts latitude and longitude to cartesian coordinates

    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees

    Returns:
        float: x, y, z coordinates
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = Re * np.cos(lat) * np.cos(lon)
    y = Re * np.cos(lat) * np.sin(lon)
    z = Re * np.sin(lat)
    return np.array([x, y, z])

def get_true_north_pointing_vector(P: np.ndarray) -> np.ndarray:
    """Get the true north pointing vector at a given point P

    Args:
        P (np.ndarray): Point P in cartesian coordinates

    Returns:
        np.ndarray: True north pointing vector
    """
    x, y, z = P
    north = np.array([-x * z, -y * z, x**2 + y**2])
    return north / np.linalg.norm(north)

def course_equation(vars, xp, yp, zp, xn, yn, zn, crs):
    xc, yc, zc = vars
    eq1 = xp * xc + yp * yc + zp * zc # Course vector is perpendicular to the OP vector
    eq2 = xn * xc + yn * yc + zn * zc - np.cos(np.deg2rad(crs)) # Course vector forms an angle of crs with the true north vector
    eq3 = xc**2 + yc**2 + zc**2 - 1 # Course vector is a unit vector
    return eq1, eq2, eq3

def rotate_vector(vector, axis, angle):
    """
    Rotates a 3D vector around another perpendicular vector by a given angle.
    
    Parameters:
    - vector: The 3D vector to be rotated (numpy array of shape (3,))
    - axis: The perpendicular vector to rotate around (numpy array of shape (3,))
    - angle: The angle of rotation in radians (float)
    
    Returns:
    - rotated_vector: The rotated vector (numpy array of shape (3,))
    """
    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    
    # Compute the components of the Rodrigues' rotation formula
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    dot_product = np.dot(axis, vector)
    cross_product = np.cross(axis, vector)
    
    # Apply the Rodrigues' rotation formula
    rotated_vector = (
        vector * cos_angle +
        cross_product * sin_angle +
        axis * dot_product * (1 - cos_angle)
    )
    
    return rotated_vector

def get_course_vector(P: np.ndarray, crs: float) -> np.ndarray:
    """Get the course vector at a given point P

    Args:
        P (np.ndarray): Point P in cartesian coordinates
        crs (float): Course in degrees

    Returns:
        np.ndarray: Course vector
    """
    north = get_true_north_pointing_vector(P)
    xp, yp, zp = P / np.linalg.norm(P)
    xn, yn, zn = north
    east = np.cross(north, P) / np.linalg.norm(np.cross(north, P))

    validation_ok = False # Flag to check if the course vector is pointing in the right direction

    if crs > 0 and crs < 180: # c should be pointing to the East
        c0 = np.cross(north, P) / np.linalg.norm(np.cross(north, P)) # initial guess is the East vector
    else:
        c0 = np.cross(P, north) / np.linalg.norm(np.cross(P, north)) # initial guess is the West vector

    n_attempts = 0
    while not validation_ok:
        n_attempts += 1

        if n_attempts > 10:
            print('Failed to find a valid course vector after 10 attempts.')
            return None

        # Solve the course equation
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                c = fsolve(course_equation, c0, args=(xp, yp, zp, xn, yn, zn, crs))
            except RuntimeWarning:
                return None

        # Normalize the course vector    
        c = c / np.linalg.norm(c)

        # Validate the course vector
        if np.dot(np.cross(north, c), P/Re) < 0: # c is pointing in the East
            if crs > 0 and crs < 180:
                validation_ok = True
            else: # 180 < crs < 360
                # c is pointing in the East, but it should be pointing in the West
                # print('Hihi E->W')
                return rotate_vector(c, P/Re, np.deg2rad(2*(360 - crs))) # 
        else: # c is pointing in the West
            if crs > 180 and crs < 360:
                validation_ok = True
            else: # 0 < crs < 180
                # c is pointing in the West, but it should be pointing in the East
                # print('Hihi W->E')
                return rotate_vector(c, -P/Re, np.deg2rad(2 * crs))

    return c

def get_second_vector_in_GC(P: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Get the second vector in the great circle defined by the course vector

    Args:
        P (np.ndarray): Point P in cartesian coordinates
        c (np.ndarray): Course vector

    Returns:
        np.ndarray: Second vector in the great circle
    """
    s = np.cross(np.cross(P, c), P)
    return s / np.linalg.norm(s)

def get_course_vector_on_GC(P: np.ndarray, s: np.ndarray, P0: np.ndarray) -> np.ndarray:
    """Get the course vector on the great circle defined by the second vector

    Args:
        P (np.ndarray): Point P in cartesian coordinates (perturbed P0)
        s (np.ndarray): Second vector in the great circle
        P0 (np.ndarray): The original point P in cartesian coordinates, to ensure the course vector is pointing in the right direction

    Returns:
        np.ndarray: Course vector on the great circle
    """
    normal_vec_of_GC = np.cross(P0, s)
    c = np.cross(normal_vec_of_GC, P)
    return c/np.linalg.norm(c)

def get_track_drift_rate(lat: float, lon: float, initial_crs: float):
    """Get the track drift rate at a given point

    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees
        initial_crs (float): Initial course in degrees

    Returns:
        drift_rate: Track drift rate in degrees per kilometer travel along the great-circle
        
    Example:
        drift_rate = get_track_drift_rate(37.7749, -122.4194, 45)
    """
    P = latlon2xyz(lat, lon) # Point P in cartesian coordinates
    c = get_course_vector(P, initial_crs) # the course vector at P
    if c is None:
        return 0
    s = get_second_vector_in_GC(P, c) # the second vector in the great circle defined by the course vector
    # s will be in the same direction as c
    
    km_for_diff = 10
    theta = km_for_diff * 1e3/Re # 100 km for differentiation, central angle
    P_prime = P * np.cos(theta) + s * Re * np.sin(theta)

    n_prime = get_true_north_pointing_vector(P_prime)
    c_prime = get_course_vector_on_GC(P_prime, s, P)
    if c_prime is None:
        return 0
    with np.errstate(invalid='raise'):
        try:
            angle = np.rad2deg(np.arccos(np.dot(n_prime, c_prime)))
        except:
            return 0
    # We must differentiate between the two possible angles: negative and positive since the dot product is symmetric (does not reveal the direction of multiplication)
    # The idea is to compute the cross product between n_prime and c_prime, if the product is in the same direction as P', then the angle is positive (<180)
    # Otherwise, the angle is negative (>180)
    crx_nc = np.cross(n_prime, c_prime)
    if np.dot(crx_nc, P_prime/Re) > 0: # the course vector is pointing to the West
        angle = 360 - angle

    delta_angle = angle - initial_crs
    # handle discontinuity at 0/360 degrees and preventing 180 degrees turn
    if delta_angle > 180: # like 10 -> 350, gives -20
        delta_angle = delta_angle - 360
    elif delta_angle < -180: # like 350 -> 10, gives +20
        delta_angle = delta_angle + 360

    d_angle = delta_angle / km_for_diff # degrees per kilometer travel
    if d_angle > 1:
        print('Caution: drift rate is greater than 1 degree per kilometer. Check the input values.')
    return d_angle

def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the Earth's surface.

    Args:
        lat1 (float): Latitude of the first point in degrees
        lon1 (float): Longitude of the first point in degrees
        lat2 (float): Latitude of the second point in degrees
        lon2 (float): Longitude of the second point in degrees

    Returns:
        float: Great circle distance between the two points in kilometers

    Example:
        distance = great_circle_distance(37.7749, -122.4194, 40.7128, -74.0060)
    """
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Earth's radius in kilometers (mean radius = 6371 km)
    R = 6371.0

    # Calculate the distance
    distance = R * c

    return distance
