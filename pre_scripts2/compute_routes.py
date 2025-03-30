import math

# Constants
EARTH_RADIUS = 6371000  # meters

# -----------------------------
# Helper Functions
# -----------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on Earth (in meters).
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS * c

def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the initial bearing (in degrees) from point A to point B.
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad)*math.sin(lat2_rad) - math.sin(lat1_rad)*math.cos(lat2_rad)*math.cos(dlon_rad)
    
    initial_bearing = math.atan2(x, y)
    bearing_deg = (math.degrees(initial_bearing) + 360) % 360
    return bearing_deg

def convert_to_xy(lat, lon, ref_lat, ref_lon):
    """
    Convert latitude and longitude to local Cartesian (x, y) coordinates in meters,
    using an equirectangular approximation around a reference point.
    """
    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    x = (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad) * EARTH_RADIUS
    y = (lat_rad - ref_lat_rad) * EARTH_RADIUS
    return (x, y)

# -----------------------------
# Projection Fitness Function
# -----------------------------

def projection_fitness(flight_segment, edge, graph, is_endpoint=False):
    """
    Compute a fitness score for how well a flight segment (straight line between 
    recorded coordinates) matches a given edge in the network.
    
    The score is a weighted sum of:
      - Orientation (bearing) similarity.
      - Overlap (projected along the edge) between the flight segment and the edge.
    
    For the first and last segments (is_endpoint=True) the overlap is downweighted.
    """
    # Retrieve flight segment endpoints
    f_from_lat = flight_segment['from_lat']
    f_from_lon = flight_segment['from_lon']
    f_to_lat = flight_segment['to_lat']
    f_to_lon = flight_segment['to_lon']
    
    # Compute flight segment bearing and length
    flight_bearing = compute_bearing(f_from_lat, f_from_lon, f_to_lat, f_to_lon)
    flight_length = haversine_distance(f_from_lat, f_from_lon, f_to_lat, f_to_lon)
    
    # Retrieve edge endpoints from the graph nodes
    edge_from_node = graph['nodes'][edge['from']]
    edge_to_node = graph['nodes'][edge['to']]
    e_from_lat, e_from_lon = edge_from_node
    e_to_lat, e_to_lon = edge_to_node
    
    # Compute edge bearing and length
    edge_bearing = compute_bearing(e_from_lat, e_from_lon, e_to_lat, e_to_lon)
    edge_length = haversine_distance(e_from_lat, e_from_lon, e_to_lat, e_to_lon)
    
    # 1. Orientation Score:
    # Compute the minimal difference in bearing (0 means same direction)
    bearing_diff = abs(flight_bearing - edge_bearing)
    if bearing_diff > 180:
        bearing_diff = 360 - bearing_diff
    # Use a linear score: 1 if difference 0, 0 if difference >= 90 degrees.
    orientation_score = max(0, 1 - (bearing_diff / 90))
    
    # 2. Overlap Score:
    # Convert flight segment endpoints to local x,y coordinates using edge's start as reference.
    E1 = convert_to_xy(e_from_lat, e_from_lon, e_from_lat, e_from_lon)
    E2 = convert_to_xy(e_to_lat, e_to_lon, e_from_lat, e_from_lon)
    F1 = convert_to_xy(f_from_lat, f_from_lon, e_from_lat, e_from_lon)
    F2 = convert_to_xy(f_to_lat, f_to_lon, e_from_lat, e_from_lon)
    
    # Compute edge vector and its length (in meters)
    edge_vec = (E2[0] - E1[0], E2[1] - E1[1])
    L = math.hypot(edge_vec[0], edge_vec[1])
    if L == 0:
        # Degenerate edge: return a low score.
        return 0.0
    # Unit vector in edge direction.
    u = (edge_vec[0] / L, edge_vec[1] / L)
    
    # Project flight endpoints onto the edge vector (using dot product)
    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1]
    
    t1 = dot((F1[0] - E1[0], F1[1] - E1[1]), u)
    t2 = dot((F2[0] - E1[0], F2[1] - E1[1]), u)
    
    # Flight segment projected interval (in meters along edge)
    seg_start = min(t1, t2)
    seg_end = max(t1, t2)
    
    # Overlap with edge interval [0, L]
    overlap_start = max(0, seg_start)
    overlap_end = min(L, seg_end)
    overlap_length = max(0, overlap_end - overlap_start)
    
    # Normalize overlap by flight segment length (avoid division by zero)
    normalized_overlap = overlap_length / (flight_length + 1e-6)
    # Cap the normalized overlap to 1.
    normalized_overlap = min(normalized_overlap, 1.0)
    
    # For endpoints, we give a lower weight to overlap.
    if is_endpoint:
        weight_orientation = 0.8
        weight_overlap = 0.2
    else:
        weight_orientation = 0.5
        weight_overlap = 0.5
        
    fitness = weight_orientation * orientation_score + weight_overlap * normalized_overlap
    return fitness

# -----------------------------
# Subgraph Extraction Function
# -----------------------------

def get_relevant_subgraph(graph, segments, margin=0.5):
    """
    Extract the subgraph that covers the area spanned by the flight segments,
    plus a margin (in degrees).
    """
    # Compute bounding box from all segment endpoints.
    all_lats = []
    all_lons = []
    for seg in segments:
        all_lats.extend([seg['from_lat'], seg['to_lat']])
        all_lons.extend([seg['from_lon'], seg['to_lon']])
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # Apply margin.
    min_lat -= margin
    max_lat += margin
    min_lon -= margin
    max_lon += margin
    
    # Filter nodes.
    sub_nodes = {}
    for node_id, (lat, lon) in graph['nodes'].items():
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            sub_nodes[node_id] = (lat, lon)
    
    # Filter edges: include an edge if both endpoints are in sub_nodes.
    sub_edges = []
    for edge in graph['edges']:
        if edge['from'] in sub_nodes and edge['to'] in sub_nodes:
            sub_edges.append(edge)
    
    return {"nodes": sub_nodes, "edges": sub_edges}

# -----------------------------
# DFS Search for Route Inference
# -----------------------------

def dfs_search_routes(segment_index, current_route, current_fitness, segments, graph, candidate_routes, max_routes):
    """
    Recursively search for routes matching the flight segments.
    
    - segment_index: index of the current flight segment to match.
    - current_route: list of edges chosen so far.
    - current_fitness: cumulative fitness score so far.
    - segments: list of flight segments.
    - graph: the (sub)graph to search.
    - candidate_routes: list to collect complete routes (each as (route, fitness)).
    - max_routes: maximum number of candidate routes to gather.
    """
    if segment_index >= len(segments):
        # Completed a candidate route.
        candidate_routes.append((list(current_route), current_fitness))
        return
    
    current_segment = segments[segment_index]
    
    candidate_edges = []
    # For the first segment, consider all edges in the subgraph.
    if not current_route:
        for edge in graph['edges']:
            # Use a lower overlap weight for endpoints.
            fitness = projection_fitness(current_segment, edge, graph, is_endpoint=True)
            # Only consider edges scoring above a threshold (tune as needed)
            if fitness > 0.3:
                candidate_edges.append((edge, fitness))
        # Sort candidates by fitness (descending)
        candidate_edges.sort(key=lambda x: x[1], reverse=True)
    else:
        # For subsequent segments, consider outgoing edges from the last edge.
        last_edge = current_route[-1]
        last_to = last_edge['to']
        for edge in graph['edges']:
            if edge['from'] == last_to:
                fitness = projection_fitness(current_segment, edge, graph, is_endpoint=False)
                if fitness > 0.3:
                    candidate_edges.append((edge, fitness))
        # Additionally, allow a “stay on” option (if the same edge might continue to cover multiple segments)
        # This is useful when the data suggests the flight is still on the same air route edge.
        same_edge = last_edge
        fitness = projection_fitness(current_segment, same_edge, graph, is_endpoint=False)
        if fitness > 0.3:
            candidate_edges.append((same_edge, fitness))
        candidate_edges.sort(key=lambda x: x[1], reverse=True)
    
    # Expand DFS for each candidate edge.
    for edge, fitness in candidate_edges:
        new_route = current_route + [edge]
        new_fitness = current_fitness + fitness
        dfs_search_routes(segment_index + 1, new_route, new_fitness, segments, graph, candidate_routes, max_routes)
        if len(candidate_routes) >= max_routes:
            return  # Stop early if we have reached our maximum

# -----------------------------
# Main Function to Infer Routes
# -----------------------------

def infer_flight_routes(segments, graph, max_routes=5):
    """
    Given a list of flight segments and the full air route network graph, infer the candidate routes.
    
    Returns a list of tuples (route, total_fitness), where route is a list of edge dicts.
    """
    # First, restrict the graph to the relevant area.
    subgraph = get_relevant_subgraph(graph, segments, margin=0.5)
    
    candidate_routes = []
    dfs_search_routes(0, [], 0.0, segments, subgraph, candidate_routes, max_routes)
    
    # Sort candidate routes by fitness score (highest first)
    candidate_routes.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N routes (if available)
    return candidate_routes[:max_routes]

# -----------------------------
# Example Usage
# -----------------------------

if __name__ == '__main__':
    # Example flight segments (normally loaded from CSV)
    segments = [
        {
            "id": "000042HMJ225",
            "from_time": 1680349319.0,
            "to_time": 1680364679.0,
            "from_lat": 42.94166564941406,
            "from_lon": 14.271751226380816,
            "to_lat": 46.17704772949219,
            "to_lon": 14.543553794302593,
            "from_alt": 11521.44,
            "to_alt": 1569.72,
            "from_speed": 0.232055441288291,
            "to_speed": 0.1399177216897295
        },
        {
            "id": "000042HMJ225",
            "from_time": 1680364679.0,
            "to_time": 1680382679.0,
            "from_lat": 46.17704772949219,
            "from_lon": 14.543553794302593,
            "to_lat": 35.847457627118644,
            "to_lon": 14.489973352310503,
            "from_alt": 1569.72,
            "to_alt": 114.3,
            "from_speed": 0.1399177216897295,
            "to_speed": 0.0
        }
    ]
    
    # Example graph – in practice this will be much larger.
    # Here we define a few nodes and edges (the waypoint names are placeholders).
    graph = {
        "nodes": {
            "MEGAN": (42.0, 14.0),
            "TIRSA": (46.0, 14.5),
            "OTHER": (36.0, 14.5)
        },
        "edges": [
            {"id": "edge1", "from": "MEGAN", "to": "TIRSA"},
            {"id": "edge2", "from": "TIRSA", "to": "OTHER"}
        ]
    }
    
    # Infer candidate routes (up to 5)
    candidate_routes = infer_flight_routes(segments, graph, max_routes=5)
    
    # Print out the candidate routes and their total fitness scores.
    for idx, (route, fitness) in enumerate(candidate_routes):
        edge_ids = [edge.get("id", f"{edge['from']}->{edge['to']}") for edge in route]
        print(f"Route {idx+1}: Edges: {edge_ids}, Total Fitness: {fitness:.3f}")