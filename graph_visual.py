import torch
import networkx as nx
import matplotlib.pyplot as plt

# Define all gates with their adjacency matrices and example weights
gates = {
    "FALSE": torch.zeros((6,6), dtype=torch.int32),
    "TRUE": torch.zeros((6,6), dtype=torch.int32),
    "AND": torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
    "OR": torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
    "NAND": torch.tensor([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
    "NOR": torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
    "XOR": torch.tensor([
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32)
}

# Example: uniform dummy weights for visualization
dummy_weights = torch.ones((6,6)) * 10
node_labels = {0:'x0',1:'x1',2:'h2',3:'h3',4:'h4',5:'y'}

# Plot all gates
fig, axes = plt.subplots(2, 4, figsize=(20,10))
axes = axes.flatten()

for i, (name, adj) in enumerate(gates.items()):
    G = nx.DiGraph()
    # Add nodes
    for n in range(6):
        G.add_node(n, label=node_labels[n])
    # Add edges
    rows, cols = torch.nonzero(adj, as_tuple=True)
    edges = list(zip(rows.tolist(), cols.tolist()))
    edge_weights = [dummy_weights[r,c].item() for r,c in edges]
    edge_widths = [(w / max(edge_weights) * 5) if edge_weights else 1 for w in edge_weights]
    edge_colors = ['green']*len(edges)  # just uniform for simplicity
    
    for (r,c) in edges:
        G.add_edge(r,c)
    
    # Node activity
    node_activity = adj.sum(dim=1).tolist()
    node_sizes = [50 + a*800 for a in node_activity]
    
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=node_sizes,
            node_color='lightblue', width=edge_widths, edge_color=edge_colors,
            arrowsize=20, ax=axes[i])
    axes[i].set_title(name)

# Hide unused subplot if any
for j in range(i+1, len(axes)): # type: ignore
    axes[j].axis('off')

plt.suptitle("Minimal DAGs for Basic Boolean Gates")
plt.show()
