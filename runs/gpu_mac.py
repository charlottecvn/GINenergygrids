import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")

print("Is CUDA enabled?", torch.cuda.is_available())
print("Is MPS enabled?", torch.backends.mps.is_available())
print("Is CPU enabled?", torch.cpu.is_available())
print("Using device:", device)

import torch
import math
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())


# test pytorch MPS acceleration.
dtype = torch.float
device = torch.device("mps")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

# Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


import torch
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

device = torch.device("mps")

### Import data
dataset = Planetoid(root="tmp/Cora", name="Cora")
data = dataset[0]
print("Cora dataset:", data)

n2v_model = Node2Vec(edge_index=data.edge_index, embedding_dim=128, walk_length=10, 
                    context_size=10, walks_per_node=10, p=1, q=1, sparse=True).to(device)

def create_loader(model):
    return model.loader(batch_size=64, shuffle=True, num_workers=0)

optimizer = torch.optim.SparseAdam(list(n2v_model.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

if __name__ == '__main__':
    loader = create_loader(n2v_model)
    loss=[]
    n2v_model.train()
    for sample in loader:
        print(sample)
        optimizer.zero_grad()
        break