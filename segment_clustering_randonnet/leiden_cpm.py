from collections import defaultdict
import random

class Graph:
    def __init__(self):
        self.adj = defaultdict(list)  # Changed to list to accommodate weighted edges
        self.nodes = set()
        self.total_weight = 0

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))
        self.nodes.update([u, v])
        self.total_weight += weight * 2  # since it's undirected

    def neighbors(self, u):
        return self.adj[u]

class LeidenCPM:
    def __init__(self, graph, gamma=1.0):
        self.graph = graph
        self.gamma = gamma  # Resolution parameter for CPM
        self.partition = {node: i for i, node in enumerate(graph.nodes)}
        self.communities = defaultdict(set)
        for node, comm in self.partition.items():
            self.communities[comm].add(node)
        self.quality = self.calculate_quality()

    def calculate_quality(self):
        # Calculate CPM quality based on current partition
        quality = 0.0
        for comm, nodes in self.communities.items():
            internal_weight = 0
            for node in nodes:
                for neighbor, weight in self.graph.neighbors(node):
                    if self.partition[neighbor] == comm:
                        internal_weight += weight
            internal_weight /= 2  # Each internal edge is counted twice
            size = len(nodes)
            quality += internal_weight - self.gamma * (size * (size - 1)) / 2
        return quality

    def run(self):
        improvement = True
        while improvement:
            improvement = self.one_level()
            self.quality = self.calculate_quality()
            self.refinement()
            self.quality = self.calculate_quality()
            # Aggregation step can be added here for multi-level optimization
        return self.partition

    def one_level(self):
        # Perform one level of local moving
        improved = False
        nodes = list(self.graph.nodes)
        random.shuffle(nodes)
        for node in nodes:
            current_comm = self.partition[node]
            # Calculate the best community for the node
            best_comm, best_gain = self.find_best_community(node)
            if best_comm != current_comm:
                self.move_node(node, current_comm, best_comm)
                improved = True
        return improved

    def find_best_community(self, node):
        # Calculate the gain for moving the node to each neighboring community
        community_gains = defaultdict(float)
        node_degree = sum(weight for _, weight in self.graph.neighbors(node))
        internal_edges = defaultdict(float)

        for neighbor, weight in self.graph.neighbors(node):
            neighbor_comm = self.partition[neighbor]
            internal_edges[neighbor_comm] += weight

        # Current community before moving
        current_comm = self.partition[node]

        # Calculate the gain for each neighboring community
        for comm, w_in in internal_edges.items():
            if comm == current_comm:
                continue
            size = len(self.communities[comm])
            gain = w_in - self.gamma * size
            community_gains[comm] += gain

        # Also consider staying in the current community
        size_current = len(self.communities[current_comm])
        # Removing node from current community
        internal_current = internal_edges.get(current_comm, 0)
        gain_current = -internal_current - self.gamma * (size_current - 1)
        community_gains[current_comm] += gain_current

        # Find the community with the maximum gain
        best_comm = current_comm
        best_gain = 0.0
        for comm, gain in community_gains.items():
            if gain > best_gain:
                best_gain = gain
                best_comm = comm

        return best_comm, best_gain
    
    def find_best_community_weighted(self, node): # try this later
        community_gains = defaultdict(float)
        internal_edges = defaultdict(float)

        # Calculate the sum of internal edge weights to each neighboring community
        for neighbor, weight in self.graph.neighbors(node):
            neighbor_comm = self.partition[neighbor]
            internal_edges[neighbor_comm] += weight

        current_comm = self.partition[node]

        # Calculate the gain for each neighboring community
        for comm, w_in in internal_edges.items():
            if comm == current_comm:
                continue
            # Weighted Size Factor: Sum of degrees of nodes in the community
            weighted_size = sum(
                sum(weight for _, weight in self.graph.neighbors(n))
                for n in self.communities[comm]
            )
            gain = w_in - self.gamma * weighted_size
            community_gains[comm] += gain

        # Determine the best community based on the maximum gain
        best_comm = current_comm
        best_gain = 0.0
        for comm, gain in community_gains.items():
            if gain > best_gain:
                best_gain = gain
                best_comm = comm

        return best_comm, best_gain


    def move_node(self, node, current_comm, new_comm):
        # Remove from current community
        self.communities[current_comm].remove(node)
        if not self.communities[current_comm]:
            del self.communities[current_comm]
        # Add to new community
        self.communities[new_comm].add(node)
        self.partition[node] = new_comm

    def refinement(self):
        # Ensure that each community is connected
        for comm in list(self.communities.keys()):
            subgraph = self.extract_subgraph(comm)
            components = self.find_connected_components(subgraph)
            if len(components) > 1:
                # Split the community into connected components
                del self.communities[comm]
                for component in components:
                    new_comm = max(self.partition.values()) + 1
                    for node in component:
                        self.partition[node] = new_comm
                        self.communities[new_comm].add(node)

    def extract_subgraph(self, comm):
        subgraph = Graph()
        nodes = self.communities[comm]
        for u in nodes:
            for v, w in self.graph.neighbors(u):
                if v in nodes:
                    subgraph.add_edge(u, v, w)
        return subgraph

    def find_connected_components(self, subgraph):
        visited = set()
        components = []

        def dfs(u, component):
            visited.add(u)
            component.add(u)
            for v, _ in subgraph.neighbors(u):
                if v not in visited:
                    dfs(v, component)

        for node in subgraph.nodes:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)
        return components

# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    g = Graph()
    edges = [
        (1, 2, 1), (1, 3, 1), (2, 3, 1),
        (4, 5, 1), (5, 6, 1), (4, 6, 1),
        (3, 4, 1), (3, 5, 1), (6, 7, 1),
        (7, 8, 1), (8, 9, 1), (7, 9, 1),
        (9, 10, 1), (10, 11, 1), (11, 12, 1),
        (10, 12, 1)
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    # Initialize Leiden algorithm with CPM
    gamma = 0.7  # You can adjust the resolution parameter here
    leiden = LeidenCPM(g, gamma=gamma)
    partition = leiden.run()

    # Print the communities
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    for comm, nodes in communities.items():
        print(f"Community {comm}: {sorted(nodes)}")


    # Visualize the graph
    import networkx as nx
    import matplotlib.pyplot as plt

    # Convert our Graph to a NetworkX graph for visualization
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm

    # Create a color map for communities
    color_map = plt.cm.get_cmap('tab20')
    colors = [color_map(i) for i in range(len(communities))]

    # Draw nodes with community colors
    color_index = 0 
    for comm, nodes in communities.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[colors[color_index]] * len(nodes), node_size=700)
        color_index += 1

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Add labels to nodes
    nx.draw_networkx_labels(G, pos)

    plt.title("Sample Graph Visualization with Colored Communities")
    plt.axis('off')  # Turn off axis
    plt.tight_layout()

    # Add a legend for communities
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Community {i}',
                                  markerfacecolor=color, markersize=10)
                       for i, color in enumerate(colors[:len(communities)])]
    plt.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.show()