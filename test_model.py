import torch
import torch.nn as nn
import torch.optim as optim

class SparseDAG(nn.Module):
    def __init__(self, total_nodes):
        super().__init__()
        self.N = total_nodes
        
        # Weight matrix
        self.W = nn.Parameter(torch.randn(self.N, self.N) * 0.1)
        self.b = nn.Parameter(torch.zeros(self.N))
        
        # Upper triangular mask (enforces DAG)
        mask = torch.triu(torch.ones(self.N, self.N), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        s = []

        # First two nodes are inputs
        s.append(x[0])
        s.append(x[1])

        # Compute remaining nodes
        for j in range(2, self.N):
            prev_states = torch.stack(s)

            weights = self.W[:j, j] * self.mask[:j, j] # type: ignore
            weighted_sum = (weights * prev_states).sum()

            node_value = torch.sigmoid(weighted_sum + self.b[j])
            s.append(node_value)

        return s[-1]


# XOR dataset
data = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

targets = torch.tensor([0.0, 1.0, 1.0, 0.0])

model = SparseDAG(total_nodes=6)  # 2 inputs + 3 hidden + 1 output

optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCELoss()

lambda_l1 = 1e-5  # sparsity strength
lambda_nodes = 1.35e-4

for epoch in range(5000):
    total_loss = 0
    
    for x, y in zip(data, targets):
        optimizer.zero_grad()
        
        output = model(x)
        
        loss = loss_fn(output, y)

        # edge sparsity
        l1_penalty = torch.abs(model.W * model.mask).sum() # type: ignore
        loss += lambda_l1 * l1_penalty

        # node-count penalty
        c = 10.0
        outgoing_sum = torch.sum(torch.abs(model.W) * model.mask, dim=1) # type: ignore
        node_activity = 1 - torch.exp(-c * outgoing_sum)
        node_count_penalty = node_activity.sum()
        loss += lambda_nodes * node_count_penalty
        
        # L1 penalty
        l1_penalty = torch.abs(model.W * model.mask).sum() # type: ignore
        loss = loss + lambda_l1 * l1_penalty
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

with torch.no_grad():
    W = model.W * model.mask # type: ignore
    print("Final Weights:")
    print(W.round(decimals=2))

threshold = 0.1
active_edges = (torch.abs(W) > threshold)
print(active_edges.int())

print("\nTruth Table:")
with torch.no_grad():
    for x in data:
        output = model(x)
        print(f"{int(x[0].item())} XOR {int(x[1].item())} = {output.item():.4f}")
