import networkx as nx
import torch
from torch_geometric.data import Data

# Create a small undirected social network graph
G = nx.Graph()

# Define edges: two disconnected triangles
# Nodes 0-1-2 → normal users community (triangle)
# Nodes 3-4-5 → suspicious bots cluster (triangle)
edges = [(0, 1), (1, 2), (2, 0),
         (3, 4), (4, 5), (5, 3)]

G.add_edges_from(edges)

# Explicitly add all nodes
G.add_nodes_from([0, 1, 2, 3, 4, 5])

# (6 nodes → 6-dimensional identity matrix)
num_nodes = 6
features = torch.eye(num_nodes, dtype=torch.float)  # Shape: [6, 6]

# Node labels: 0 = normal user, 1 = bot/suspicious
labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

# Convert NetworkX edges to PyG edge_index format
edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

# Make the graph undirected by adding reverse edges
# Original: [2, 6], after concatenation: [2, 12]
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

# Create PyG Data object
data = Data(
    x=features,         # Node feature matrix
    edge_index=edge_index,  # Edge list in COO format
    y=labels            # Ground-truth node labels
)

# Define train/test masks
# First 4 nodes → used for training
# Last 2 nodes  → used for testing (transductive setting)
data.train_mask = torch.tensor([True, True, True, True, False, False], dtype=torch.bool)
data.test_mask  = torch.tensor([False, False, False, False, True, True], dtype=torch.bool)

# Print to verify everything is correct
print(data)
print("Edge index:\n", data.edge_index)
print("Node features shape:", data.x.shape)

print("Labels:", data.y)