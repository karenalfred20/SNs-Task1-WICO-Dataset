# gcn_cora.py - Classic 2-layer GCN on Cora dataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load the Cora dataset (citation network)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # Only one graph in the dataset

# Print dataset info
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features: {data.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Training nodes: {data.train_mask.sum().item()}")
print(f"Test nodes: {data.test_mask.sum().item()}")

# Define a simple 2-layer GCN model
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, out_channels=None):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels or dataset.num_classes)

    def forward(self, x, edge_index):
        # First GCN layer + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        # Output log probabilities
        return F.log_softmax(x, dim=1)

# Initialize model and optimizer
model = SimpleGCN(
    in_channels=dataset.num_node_features,
    hidden_channels=16,
    out_channels=dataset.num_classes
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
model.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f}')

# Evaluation on test set
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)

correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'\nGCN Test Accuracy on Cora: {acc:.4f} ({acc*100:.2f}%)')

